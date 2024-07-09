from fastapi import FastAPI, HTTPException
from langchain_core.pydantic_v1 import BaseModel as ModelTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel as Struture
from typing import Optional, Literal
from langchain_openai import ChatOpenAI

prompt1 = PromptTemplate.from_template(
    """You will be provided with a description of face of a person. Based on the description, your task is to determine the most appropriate set of attributes from the list provided. Use your understanding and contextual clues to infer the most likely attributes. Have contextual understanding like, how a women or men face looks like, only men will have mustache and bread, so always -1 for women, etc.
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair are the attributes to be understood from description.
Task:
Using the provided description, identify the most appropriate attributes from the list. If the description is unclear or lacking specific details, use contextual understanding to infer the most likely attributes. If there is a presence of attribute, then 1 for that particular field, else -1
Description : "{text}"
"""
)
prompt2 = PromptTemplate.from_template(
    """You will be provided with a description of face of a person. Based on the description, your task is to determine the most appropriate set of attributes from the list provided. Use your understanding and contextual clues to infer the most likely attributes. Have contextual understanding like, how a women or men face looks like, only men will have mustache and bread, so always -1 for women, etc.
Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones are the attributes to be understood from description.
Task:
Using the provided description, identify the most appropriate attributes from the list. If the description is unclear or lacking specific details, use contextual understanding to infer the most likely attributes. If there is a presence of attribute, then 1 for that particular field, else -1
Description : "{text}"
"""
)

prompt3 = PromptTemplate.from_template(
    """You will be provided with a description of face of a person. Based on the description, your task is to determine the most appropriate set of attributes from the list provided. Use your understanding and contextual clues to infer the most likely attributes. Have contextual understanding like, how a women or men face looks like, only men will have mustache and bread, so always -1 for women, etc.
Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks are the attributes to be understood from description.
Task:
Using the provided description, identify the most appropriate attributes from the list. If the description is unclear or lacking specific details, use contextual understanding to infer the most likely attributes. If there is a presence of attribute, then 1 for that particular field, else -1
Description : "{text}"
"""
)

prompt4 = PromptTemplate.from_template(
    """You will be provided with a description of face of a person. Based on the description, your task is to determine the most appropriate set of attributes from the list provided. Use your understanding and contextual clues to infer the most likely attributes. Have contextual understanding like,  how a women or men face looks like, only men will have mustache and bread, so always -1 for women, etc.
Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young are the attributes to be understood from description.
Task:
Using the provided description, identify the most appropriate attributes from the list. If the description is unclear or lacking specific details, use contextual understanding to infer the most likely attributes. If there is a presence of attribute, then 1 for that particular field, else -1
Description : "{text}"
"""
)




app = FastAPI()


class InputText(Struture):
    text: str


class AttributeSet1(ModelTemplate):
    FiveOClockShadow: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a 5 o'clock shadow in the face described in the given description."
    )
    ArchedEyebrows: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of arched eyebrows in the face described in the given description."
    )
    Attractive: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of attractiveness in the face described in the given description."
    )
    BagsUnderEyes: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of bags under eyes in the face described in the given description."
    )
    Bald: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of baldness in the face described in the given description."
    )
    Bangs: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of bangs in the face described in the given description."
    )
    BigLips: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of big lips in the face described in the given description."
    )
    BigNose: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a big nose in the face described in the given description."
    )
    BlackHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of black hair in the face described in the given description."
    )
    BlondHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of blond hair in the face described in the given description."
    )

class AttributeSet2(ModelTemplate):
    Blurry: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of blurriness in the face described in the given description."
    )
    BrownHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of brown hair in the face described in the given description."
    )
    BushyEyebrows: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of bushy eyebrows in the face described in the given description."
    )
    Chubby: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of chubbiness in the face described in the given description."
    )
    DoubleChin: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a double chin in the face described in the given description."
    )
    Eyeglasses: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of eyeglasses in the face described in the given description."
    )
    Goatee: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a goatee in the face described in the given description."
    )
    GrayHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of gray hair in the face described in the given description."
    )
    HeavyMakeup: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of heavy makeup in the face described in the given description."
    )
    HighCheekbones: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of high cheekbones in the face described in the given description."
    )

class AttributeSet3(ModelTemplate):
    Male: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of maleness in the face described in the given description."
    )
    MouthSlightlyOpen: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a slightly open mouth in the face described in the given description."
    )
    Mustache: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a mustache in the face described in the given description."
    )
    NarrowEyes: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of narrow eyes in the face described in the given description."
    )
    NoBeard: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The absence of a beard in the face described in the given description."
    )
    OvalFace: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of an oval face in the face described in the given description."
    )
    PaleSkin: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of pale skin in the face described in the given description."
    )
    PointyNose: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a pointy nose in the face described in the given description."
    )
    RecedingHairline: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of a receding hairline in the face described in the given description."
    )
    RosyCheeks: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of rosy cheeks in the face described in the given description."
    )

class AttributeSet4(ModelTemplate):
    Sideburns: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of sideburns in the face described in the given description."
    )
    Smiling: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of smiling in the face described in the given description."
    )
    StraightHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of straight hair in the face described in the given description."
    )
    WavyHair: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wavy hair in the face described in the given description."
    )
    WearingEarrings: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wearing earrings in the face described in the given description."
    )
    WearingHat: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wearing a hat in the face described in the given description."
    )
    WearingLipstick: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wearing lipstick in the face described in the given description."
    )
    WearingNecklace: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wearing a necklace in the face described in the given description."
    )
    WearingNecktie: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of wearing a necktie in the face described in the given description."
    )
    Young: Optional[Literal[-1, 1]] = Field(
        ..., 
        description="The presence of youthfulness in the face described in the given description."
    )
 
 
class FaceAttributes(Struture):
    FiveOClockShadow: Optional[Literal[-1, 1]]
    ArchedEyebrows: Optional[Literal[-1, 1]]
    Attractive: Optional[Literal[-1, 1]]
    BagsUnderEyes: Optional[Literal[-1, 1]]
    Bald: Optional[Literal[-1, 1]]
    Bangs: Optional[Literal[-1, 1]]
    BigLips: Optional[Literal[-1, 1]]
    BigNose: Optional[Literal[-1, 1]]
    BlackHair: Optional[Literal[-1, 1]]
    BlondHair: Optional[Literal[-1, 1]]
    Blurry: Optional[Literal[-1, 1]]
    BrownHair: Optional[Literal[-1, 1]]
    BushyEyebrows: Optional[Literal[-1, 1]]
    Chubby: Optional[Literal[-1, 1]]
    DoubleChin: Optional[Literal[-1, 1]]
    Eyeglasses: Optional[Literal[-1, 1]]
    Goatee: Optional[Literal[-1, 1]]
    GrayHair: Optional[Literal[-1, 1]]
    HeavyMakeup: Optional[Literal[-1, 1]]
    HighCheekbones: Optional[Literal[-1, 1]]
    Male: Optional[Literal[-1, 1]]
    MouthSlightlyOpen: Optional[Literal[-1, 1]]
    Mustache: Optional[Literal[-1, 1]]
    NarrowEyes: Optional[Literal[-1, 1]]
    NoBeard: Optional[Literal[-1, 1]]
    OvalFace: Optional[Literal[-1, 1]]
    PaleSkin: Optional[Literal[-1, 1]]
    PointyNose: Optional[Literal[-1, 1]]
    RecedingHairline: Optional[Literal[-1, 1]]
    RosyCheeks: Optional[Literal[-1, 1]]
    Sideburns: Optional[Literal[-1, 1]]
    Smiling: Optional[Literal[-1, 1]]
    StraightHair: Optional[Literal[-1, 1]]
    WavyHair: Optional[Literal[-1, 1]]
    WearingEarrings: Optional[Literal[-1, 1]]
    WearingHat: Optional[Literal[-1, 1]]
    WearingLipstick: Optional[Literal[-1, 1]]
    WearingNecklace: Optional[Literal[-1, 1]]
    WearingNecktie: Optional[Literal[-1, 1]]
    Young: Optional[Literal[-1, 1]]


def combine(attributes_1, attributes_2, attributes_3, attributes_4):
    result = FaceAttributes(
        FiveOClockShadow=attributes_1.FiveOClockShadow,
        ArchedEyebrows=attributes_1.ArchedEyebrows,
        Attractive=attributes_1.Attractive,
        BagsUnderEyes=attributes_1.BagsUnderEyes,
        Bald=attributes_1.Bald,
        Bangs=attributes_1.Bangs,
        BigLips=attributes_1.BigLips,
        BigNose=attributes_1.BigNose,
        BlackHair=attributes_1.BlackHair,
        BlondHair=attributes_1.BlondHair,
        Blurry=attributes_2.Blurry,
        BrownHair=attributes_2.BrownHair,
        BushyEyebrows=attributes_2.BushyEyebrows,
        Chubby=attributes_2.Chubby,
        DoubleChin=attributes_2.DoubleChin,
        Eyeglasses=attributes_2.Eyeglasses,
        Goatee=attributes_2.Goatee,
        GrayHair=attributes_2.GrayHair,
        HeavyMakeup=attributes_2.HeavyMakeup,
        HighCheekbones=attributes_2.HighCheekbones,
        Male=attributes_3.Male,
        MouthSlightlyOpen=attributes_3.MouthSlightlyOpen,
        Mustache=attributes_3.Mustache,
        NarrowEyes=attributes_3.NarrowEyes,
        NoBeard=attributes_3.NoBeard,
        OvalFace=attributes_3.OvalFace,
        PaleSkin=attributes_3.PaleSkin,
        PointyNose=attributes_3.PointyNose,
        RecedingHairline=attributes_3.RecedingHairline,
        RosyCheeks=attributes_3.RosyCheeks,
        Sideburns=attributes_4.Sideburns,
        Smiling=attributes_4.Smiling,
        StraightHair=attributes_4.StraightHair,
        WavyHair=attributes_4.WavyHair,
        WearingEarrings=attributes_4.WearingEarrings,
        WearingHat=attributes_4.WearingHat,
        WearingLipstick=attributes_4.WearingLipstick,
        WearingNecklace=attributes_4.WearingNecklace,
        WearingNecktie=attributes_4.WearingNecktie,
        Young=attributes_4.Young,
    )
    return result

    
llm1 = ChatOpenAI(
    base_url="http://localhost:8000/v1", api_key="sk-xxx",
    temperature=0,
)
llm2 = ChatOpenAI(
    base_url="http://localhost:8001/v1", api_key="sk-xxx",
    temperature=0,
)
llm3 = ChatOpenAI(
    base_url="http://localhost:8002/v1", api_key="sk-xxx",
    temperature=0,
)
llm4 = ChatOpenAI(
    base_url="http://localhost:8003/v1", api_key="sk-xxx",
    temperature=0,
)

runnable1 = prompt1 | llm1.with_structured_output(
            schema=AttributeSet1,
            include_raw=True)
runnable2 = prompt2 | llm2.with_structured_output(
            schema=AttributeSet2,
            include_raw=True)
runnable3 = prompt3 | llm3.with_structured_output(
            schema=AttributeSet3,
            include_raw=True)
runnable4 = prompt4 | llm4.with_structured_output(
            schema=AttributeSet4,
            include_raw=True)
runnables = [runnable1, runnable2, runnable3, runnable4]

@app.post("/geolocation", response_model=FaceAttributes)
async def get_geolocation(input_text: InputText):
    try:
        attributes_1 = runnable1.invoke(input_text.text)['parsed']
        attributes_2 = runnable2.invoke(input_text.text)['parsed']
        attributes_3 = runnable3.invoke(input_text.text)['parsed']
        attributes_4 = runnable4.invoke(input_text.text)['parsed']
        attributes = combine(attributes_1, attributes_2, attributes_3, attributes_4)
        
        return attributes
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)