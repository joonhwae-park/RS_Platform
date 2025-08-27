/*
  # Add columns for new questionnaire questions

  1. New Columns
    - `openness_to_experience` (text) - 7-point Likert scale response
    - `risk_aversion` (text) - 7-point Likert scale response  
    - `movie_expertise` (text) - 7-point expertise scale response

  2. Changes
    - Added three new required columns to questionnaire_responses table
    - All columns are text type to store the scale values (1-7)
*/

-- Add new columns for the psychological measures
ALTER TABLE questionnaire_responses 
ADD COLUMN IF NOT EXISTS openness_to_experience text,
ADD COLUMN IF NOT EXISTS risk_aversion text,
ADD COLUMN IF NOT EXISTS movie_expertise text;