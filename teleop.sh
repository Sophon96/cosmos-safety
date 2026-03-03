lerobot-teleoperate.exe \
	--robot.type=so101_follower \
	--robot.port=COM3 \
	--robot.id=follower \
	--robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
	--teleop.type=so101_leader \
	--teleop.port=COM4 \
	--teleop.id=leader \
	--display_data=true
