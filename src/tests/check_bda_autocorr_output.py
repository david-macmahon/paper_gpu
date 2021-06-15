import numpy as np
import struct
import redis
import matplotlib.pyplot as plt

r = redis.Redis('redishost')

# From fake data script:
# Data for one stokes, one channel, real + imag:
# '\x00\x00\x00\x01\x00\x00\x00\x01'

int_bin = np.tile(np.arange(128, dtype=np.int32)+1, 48)

auto10n = struct.unpack('<6144I', r.get('auto:10n'))

plt.plot(auto10n)
plt.plot(int_bin, ls='--', c='g')
plt.show()
