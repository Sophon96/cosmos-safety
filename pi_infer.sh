rm -rf ~/.cache/huggingface/lerobot/Sophon96/eval_pour-milk

COSMOS_REMOTE_URL=http://127.0.0.1:8000 lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM3 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=COM4 \
    --teleop.id=leader \
    --display_data=false \
    --dataset.repo_id=Sophon96/eval_pour-milk \
    --dataset.single_task="pour milk" \
    --dataset.streaming_encoding=false \
    --dataset.vcodec="av1_qsv" \
    --dataset.encoder_threads=6 \
    --policy.path=blueplus/cosmos-reason-pi0.5 \
    --cosmos_safety.enabled=True
