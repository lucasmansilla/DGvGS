import hashlib


def get_seed_hash(*args):
    m = hashlib.md5(str(args).encode('utf-8'))
    h = m.hexdigest()
    i = int(h, 16)
    seed = i % (2**31)
    return seed
