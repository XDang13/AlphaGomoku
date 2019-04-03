from functools import wraps

def singleton(cls, *args, **kw):
    instance={}
    def _singleton():
        if cls not in instance:
            instance[cls]=cls(*args, **kw)
        return instance[cls]
    return _singleton
    
@singleton
class Black():
    name = "black"
    value = 1
    @property
    def oppent(self):
        return White()

@singleton
class White():
    name = "white"
    value = -1
    @property
    def oppent(self):
        return Black()