"""Official float32 Edge/CLIP forward with ID 8 and frozen literal instructions."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import edge_isaac_service as base
from research.isaac_final_common import append_json,sha256,write_json
from research.isaac_final_contracts import action,pixel_hash
from research.pose_language_contracts import PLAN,FLAGS,MASK,source_masks,tensor_hash,instruction_fields,validate_instruction,verify_protocol


def prepare(frames,normalized_pose,spec):
    batch=base.prepare(frames,normalized_pose)
    tokens=base.clip.tokenize(spec['text'],truncate=False)
    assert tokens.numpy().tolist()==spec['token_ids']
    assert tensor_hash(tokens.numpy())==spec['token_sha256']
    batch['obj_inst_lan']=tokens.to(base.official.device)
    return batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['offline','navigation','infrastructure'],required=True)
    args=p.parse_args()
    dummy=args.phase=='infrastructure'
    if dummy: plan=json.loads(PLAN.read_text())
    else: _,plan,_=verify_protocol()
    args.output.mkdir(parents=True,exist_ok=False)
    hook={};torch=base.torch
    if not dummy:
        forward=base.load(args.output)  # strict original loader, no forward/warmup
        for name,value in FLAGS.items():setattr(base.official,name,value)
        masks=source_masks(torch)
        assert torch.equal(base.official.model.all_masks,masks.all_masks)
        assert torch.equal(base.official.model.avg_pool_mask,masks.avg_pool_mask)
        write_json(args.output/'modality-gate.json',dict(status='MODALITY8_INPUT_VALID',flags=FLAGS,
            mask=MASK,pool_weights=masks.avg_pool_mask[8].tolist(),source_selection_id=8))

        def before_model(module,inputs):
            assert inputs[4].tolist()==[8]
            embedding=inputs[5].detach().cpu().numpy()
            assert embedding.shape==(1,512) and embedding.dtype==np.float32 and np.isfinite(embedding).all()
            hook.update(clip_embedding_sha256=tensor_hash(embedding),clip_embedding_shape=list(embedding.shape),
                clip_embedding_dtype=str(embedding.dtype),actual_model_modality=8,
                actual_goal_float32=inputs[1].detach().cpu().numpy().tolist())
            torch.cuda.synchronize();hook['model_start']=time.perf_counter()

        def before_decoder(module,inputs):
            assert inputs[1].tolist()==[MASK] and inputs[3].tolist()==[8]
            weights=inputs[2][8].detach().cpu().tolist()
            assert weights==[0. if m else 1.25 for m in MASK]
            hook.update(decoder_mask=MASK,decoder_pool_weights=weights)

        def after_model(*_):
            torch.cuda.synchronize();hook['model_ms']=(time.perf_counter()-hook['model_start'])*1000

        base.official.model.register_forward_pre_hook(before_model)
        base.official.model.decoder.register_forward_pre_hook(before_decoder)
        base.official.model.register_forward_hook(after_model)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(args.socket);os.chmod(args.socket,0o600);server.listen(1);server.settimeout(.2)
    write_json(args.output/'ready.json',dict(status='READY',pid=os.getpid(),phase=args.phase,model_loaded=not dummy))
    seen=set();calls=0;attempts=0;active_episode=None;episode_index=-1;expected_request=0
    try:
        while not (args.output.parent/'service-stop').exists():
            try:conn,_=server.accept()
            except socket.timeout:continue
            with conn,conn.makefile('rb') as stream:
                for line in stream:
                    request=json.loads(line);received=time.time();start=time.perf_counter()
                    response=dict(request_id=request['request_id'],observation_id=request['observation_id'],
                        request_episode_id=request['episode_id'])
                    try:
                        spec=validate_instruction(request,plan)
                        key=(request['episode_id'],request['request_id']);assert key not in seen,'No replay'
                        seen.add(key)
                        if args.phase=='offline':
                            expected=plan['offline_order'][attempts]
                            assert all(request[k]==expected[k] for k in ('episode_id','request_id','instruction_id'))
                        elif args.phase=='navigation':
                            if request['episode_id']!=active_episode:
                                episode_index+=1;expected_request=0;active_episode=request['episode_id']
                            expected=plan['navigation_order'][episode_index]
                            assert all(request[k]==expected[k] for k in ('episode_id','instruction_id'))
                            assert request['request_id']==expected_request;expected_request+=1
                        attempts+=1
                        frames=base.verify_frames(request)
                        response.update(**instruction_fields(spec),rgb_hashes=[pixel_hash(f) for f in frames])
                        if dummy:
                            assert request['op']=='dummy'
                            response.update(dummy=True,lossless_roundtrip=True,clip_embedding_sha256=None)
                        else:
                            assert request['op']=='infer'
                            batch=prepare(frames,request['normalized_pose'],spec)
                            torch.cuda.synchronize();forward_start=time.perf_counter();hook.clear()
                            tensor,modality=forward(None,model=base.official.model,batch=batch,
                                device_id=base.official.device,mode='vali',idrun=calls)
                            torch.cuda.synchronize();forward_ms=(time.perf_counter()-forward_start)*1000
                            assert int(modality.item())==8 and hook['actual_model_modality']==8
                            assert hook['actual_goal_float32']==np.asarray([request['normalized_pose']],dtype=np.float32).tolist()
                            raw=action(tensor.detach().cpu().numpy())
                            path=Path(request['action_path']).resolve();assert path.is_relative_to(ROOT/'outputs')
                            path.parent.mkdir(parents=True,exist_ok=True)
                            with path.open('xb') as dest:
                                np.save(dest,raw,allow_pickle=False);dest.flush();os.fsync(dest.fileno())
                            path.chmod(0o444)
                            metric=raw[0,4].copy();metric[:2]*=.1
                            try:response['controller']=base.controller(metric)
                            except NameError as exc:
                                if 'clip_angle' not in str(exc):raise
                                response['controller_error']='UPSTREAM_CONTROLLER_BUG: '+str(exc)
                            response.update(action_path=str(path),action_sha256=sha256(path),native_shape=list(raw.shape),
                                native_dtype=str(raw.dtype),selected_index=4,selected_native=raw[0,4].tolist(),
                                selected_metric=metric.tolist(),forward_with_clip_ms=forward_ms,
                                **{k:v for k,v in hook.items() if k!='model_start'})
                            calls+=1
                        response.update(pipeline_ms=(time.perf_counter()-start)*1000,
                            received_wall_time=received,ready_wall_time=time.time())
                    except Exception:response['error']=traceback.format_exc()
                    append_json(args.output/'responses.jsonl',response)
                    conn.sendall((json.dumps(response,allow_nan=False)+'\n').encode())
    finally:
        server.close();Path(args.socket).unlink(missing_ok=True)
        write_json(args.output/'stopped.json',dict(inference_count=calls,request_attempts=attempts,
            load_count=0 if dummy else 1,phase=args.phase))


if __name__=='__main__':main()
