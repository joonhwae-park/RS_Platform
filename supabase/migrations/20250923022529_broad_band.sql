/*
  # Create recommendations table

  1. New Tables
    - `recommendations`
      - `id` (uuid, primary key)
      - `session_id` (uuid, foreign key to user_sessions)
      - `movie_id` (integer)
      - `score` (real)
      - `model` (text) - which model generated this recommendation
      - `phase` (integer) - which phase this recommendation is for
      - `rank` (integer) - internal ranking within the model
      - `display_order` (integer, nullable) - order for display to user
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on recommendations table
    - Add policies for public access
*/

-- Create recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  movie_id integer NOT NULL,
  score real NOT NULL,
  model text NOT NULL,
  phase integer NOT NULL DEFAULT 2,
  rank integer NOT NULL,
  display_order integer,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Enable Row Level Security
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies for recommendations
CREATE POLICY "Users can read recommendations for their sessions"
  ON recommendations
  FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Users can insert recommendations for their sessions"
  ON recommendations
  FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Users can update recommendations for their sessions"
  ON recommendations
  FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);