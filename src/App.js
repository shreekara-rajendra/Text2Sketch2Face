import Card from './Card';
import RadioButtons from './RadioButtons';
import {useState, setState, useEffect} from 'react';

import styled from 'styled-components';
function App() {
  let [counter, setCounter] = useState(0);
  let [values, setValues] = useState([]);
  const SketchedText = styled.div`
  font-size: 2em;
  font-weight: bold;
  width: 100%;
  margin-top: 4rem;
  display: flex;
  justify-content: center;
  color: black;
  text-shadow:
    0.05em 0.05em 0 rgba(0,0,0,0.2),
    -0.05em -0.05em 0 rgba(0,0,0,0.2),
    0.05em -0.05em 0 rgba(0,0,0,0.2),
    -0.05em 0.05em 0 rgba(0,0,0,0.2);
  font-family: 'Permanent Marker', cursive; /* Use a font that looks like handwriting */
`;

  let questions = [
    "Do they have a 5oClock Shadow?",
   "Do they have Arched Eyebrows?",
   "Are they Attractive?",
   "Do they have Bags Under Eyes?",
   "Are they Bald?",
   "Do they have Bangs?",
   "Do they have Big Lips?",
   "Do they have a Big Nose?",
   "Do they have Black Hair?",
   "Do they have Blond Hair?",
   "Is the photo Blurry?",
   "Do they have Brown Hair?",
   "Do they have Bushy Eyebrows?",
   "Are they Chubby?",
   "Do they have a Double Chin?",
   "Do they have Eyeglasses?",
   "Do they have a Goatee?",
   "Do they have Gray Hair?",
   "Do they have Heavy Makeup on?",
   "Do they have High Cheekbones?",
   "Are they male?",
   "Is thier mouth slightly Mouth Slightly Open?",
   "Do they have a Mustache?",
   "Do they have Narrow Eyes?",
   "Do they have No Beard?",
   "Do they have an Oval Face?",
   "Do they have Pale Skin?",
   "Do they have a Pointy Nose?",
   "Do they have a Receding Hairline?",
   "Do the have Rosy Cheeks?",
   "Do they have Sideburns?",
   "Are they Smiling?",
   "Do they have Straight Hair?",
   "Do they have Wavy Hair?",
   "Are they Wearing Earrings?",
   "Are they Wearing Hat?",
   "Are they Wearing Lipstick?",
   "Are they Wearing Necklace?",
   "Are they Wearing Necktie?",
   "Are they Young?"
];
  function handleSubmit(value) {  
    values.push(value);
    setCounter(counter+1);
  }
  function submit() {

  }
  useEffect(
    ()=> {
      if(counter==40) {
        submit();
        alert(values);
      }
    }
  )
  return (
    <div
   

    >
      <SketchedText>
        Text2Sketch2Face
      </SketchedText>
       <Card style={{display:'flex', justifyContent:'center', alignItems:'center'}}>
     

     <div style={{display:'flex', flexDirection:'column', justifycontent:'space-between', alignItems:'center', width: '50vw', height: '25vh'}}>
     <h2>
               {questions[counter]}
             </h2> 
             <div>
           <RadioButtons callback={handleSubmit}/>
           </div>
     </div>
         
   
  
   </Card>
      </div>
   
  );
}

export default App;
