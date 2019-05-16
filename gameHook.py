# Test trying to connect to an already started game

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import point_flag
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

