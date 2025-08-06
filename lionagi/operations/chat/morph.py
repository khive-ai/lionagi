from ..morphism import Morphism, MorphismContext, MorphismMeta
from dataclasses import dataclass

_CHAT_MORPH_META = MorphismMeta(
    name="predict",
    version="1.0",
    description="Transform chat messages into predictions as an assistant response.",
    can_stream=True,
    stream_handler=None,
)



@dataclass(slots=True, frozen=True, init=False)
class Predict(Morphism):
    """A transformation that occurs in a space"""

    meta = _CHAT_MORPH_META
    
    





    
    
    
    
    
    
    ...






