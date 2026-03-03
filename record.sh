rm -rf ~/.cache/huggingface/lerobot/Sophon96/record-pour-milk

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM3 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=COM4 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=Sophon96/record-pour-milk \
    --dataset.num_episodes=50 \
    --dataset.single_task="pour milk" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=h264_qsv \
    --dataset.encoder_threads=2
