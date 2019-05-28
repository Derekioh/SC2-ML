# Test trying to connect to an already started game

from absl import app
from absl import flags

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import point_flag
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "AutomatonLE", "Name of a map to use to play.")

def main(unused_args):
	run_config = run_configs.get()

	#map_inst = maps.get(FLAGS.map)

	interface = sc_pb.InterfaceOptions()
	interface.raw = False
	interface.score = False
	# interface.feature_layer.width = 24
	# FLAGS.feature_screen_size.assign_to(interface.feature_layer.resolution)
	# FLAGS.feature_minimap_size.assign_to(interface.feature_layer.minimap_resolution)
	# FLAGS.rgb_screen_size.assign_to(interface.render.resolution)
	# FLAGS.rgb_minimap_size.assign_to(interface.render.minimap_resolution)

	# Starcraft 2 offical ports: 	1119, 6113, 1120, 80, 3724
	join = sc_pb.RequestJoinGame(observed_player_id=1,server_ports=sc_pb.PortSet(game_port=1119,base_port=1119),client_ports=[sc_pb.PortSet(game_port=1119,base_port=1119),sc_pb.PortSet(game_port=6113,base_port=6113),sc_pb.PortSet(game_port=1120,base_port=1120),sc_pb.PortSet(game_port=80,base_port=80),sc_pb.PortSet(game_port=3724,base_port=3724)],options=interface)
	#join = sc_pb.RequestJoinGame(observed_player_id=1,options=interface)

	try:
		with run_config.start(version=None) as controller: #version="4.8.6"
			print("SC2 Started successfully.")
			ping = controller.ping()
			print(ping)
			controller.join_game(join)

			obs = controller.observe()
			# holds our mineral, vespene gas, and food data
			playerData = obs.observation.player_common

			print("Player Data")
			print(playerData)

	except (protocol.ConnectionError, protocol.ProtocolError,
	      remote_controller.RequestError) as e:
		print(e)

	while True:
		try:
			pass
		except KeyboardInterrupt:
			break

if __name__ == "__main__":
	app.run(main)