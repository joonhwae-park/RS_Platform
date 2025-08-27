/*
  # Add new tracking tables and update existing schema

  1. New Tables
    - `phase_transitions` - Track when users move between phases
    - `trailer_views` - Track trailer watching behavior
    - `questionnaire_responses` - Store final questionnaire data
  
  2. Schema Updates
    - Add screen_width and screen_height to user_sessions
    - Remove mouse_events table (no longer needed)
  
  3. Security
    - Enable RLS on all new tables
    - Add appropriate policies
*/

-- Add screen size columns to user_sessions
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'user_sessions' AND column_name = 'screen_width'
  ) THEN
    ALTER TABLE user_sessions ADD COLUMN screen_width integer;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'user_sessions' AND column_name = 'screen_height'
  ) THEN
    ALTER TABLE user_sessions ADD COLUMN screen_height integer;
  END IF;
END $$;

-- Create phase_transitions table
CREATE TABLE IF NOT EXISTS phase_transitions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  from_phase text NOT NULL,
  to_phase text NOT NULL,
  timestamp timestamptz NOT NULL,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Create trailer_views table
CREATE TABLE IF NOT EXISTS trailer_views (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  movie_id integer NOT NULL,
  start_time timestamptz NOT NULL,
  end_time timestamptz,
  duration integer, -- in seconds
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Create questionnaire_responses table
CREATE TABLE IF NOT EXISTS questionnaire_responses (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES user_sessions(id) NOT NULL,
  movie_watching_frequency text NOT NULL,
  streaming_services text[] NOT NULL,
  primary_streaming_service text NOT NULL,
  movie_genre_preferences text[] NOT NULL,
  gender text,
  age_range text,
  nationality text,
  occupation text,
  additional_comments text,
  created_at timestamptz DEFAULT now() NOT NULL
);

-- Enable Row Level Security
ALTER TABLE phase_transitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trailer_views ENABLE ROW LEVEL SECURITY;
ALTER TABLE questionnaire_responses ENABLE ROW LEVEL SECURITY;

-- Create policies for phase_transitions
CREATE POLICY "Users can manage phase transitions for their sessions"
  ON phase_transitions
  FOR ALL
  TO public
  USING (true);

-- Create policies for trailer_views
CREATE POLICY "Users can manage trailer views for their sessions"
  ON trailer_views
  FOR ALL
  TO public
  USING (true);

-- Create policies for questionnaire_responses
CREATE POLICY "Users can manage questionnaire responses for their sessions"
  ON questionnaire_responses
  FOR ALL
  TO public
  USING (true);

-- Drop mouse_events table if it exists (no longer needed)
DROP TABLE IF EXISTS mouse_events;