import livestreamer

url = 'https://www.twitch.tv/belgrindil'

CLIENT_ID_TWITCH_WEBPLAYER = "jzkbprff40iqj646a697cyrvl0zt2m6"

session = livestreamer.Livestreamer()
session.set_option("http-headers","Client-ID=" + CLIENT_ID_TWITCH_WEBPLAYER)
#URL = "https://api.twitch.tv/api/channels/" + streamName + "?client_id=" + CLIENT_ID_TWITCH_WEBPLAYER
#streams = session.streams(URL)
#stream = streams['best']

plugin = session.resolve_url("http://twitch.tv/" + "belgrindil")
streams = plugin.get_streams()
#stream = streams['best']
stream = streams['1080p60']

data = bytearray()
fd = stream.open()
i = 0
while i < 5:
	data = data + fd.read(4000024)
	i += 1
fd.close()
fname = 'stream3.mp4'
f = open(fname, 'wb')
f.write(data)
f.close()