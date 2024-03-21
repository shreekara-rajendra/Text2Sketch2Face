import React, { useState } from 'react';
import styled from 'styled-components';

const StyledButton = styled.button`
  background-color: black; /* Green background */
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 1.125rem;;
  transition: background-color 0.3s ease;

  &:hover {
    background-color: grey; /* Darker green background on hover */
  }
`;
function YesNoRadioButtons({callback}) {


  const handleChange = (event) => {
  
    callback(event.target.value);
  };

  return (
    <div>
      <StyledButton onClick={(e)=> {callback(1);}}>  Yes</StyledButton>
      <StyledButton onClick={(e)=> {callback(-1);}}> No</StyledButton>
    </div>
  );
}

export default YesNoRadioButtons;
