/*
  # Create Phase 1 Movie Selection Function

  1. Function
    - `select_phase1_movies_for_session(session_id uuid)` 
    - Returns array of movie_ids (text[])
    
  2. Logic
    - Step 1: Select 10 random movies from top_vote_count = 1
    - Step 2: Exclude Step 1 movies, select 10 random from top_vote_variance = 1
    - Step 3: For each genre (1-10), select 1 random movie from top_genre = N
    - Store selections in session_phase1_movies table
    - Return array of 30 movie IDs
    
  3. Notes
    - Function is idempotent - if session already has Phase 1 movies, return existing
    - Uses true randomness with ORDER BY random()
    - Ensures no overlap between steps
*/

CREATE OR REPLACE FUNCTION select_phase1_movies_for_session(p_session_id uuid)
RETURNS text[]
LANGUAGE plpgsql
AS $$
DECLARE
  v_existing_count integer;
  v_vote_count_movies text[];
  v_vote_variance_movies text[];
  v_genre_movies text[];
  v_all_movies text[];
  v_movie_id text;
  v_genre_num integer;
BEGIN
  -- Check if session already has Phase 1 movies
  SELECT COUNT(*) INTO v_existing_count
  FROM session_phase1_movies
  WHERE session_id = p_session_id;
  
  IF v_existing_count > 0 THEN
    -- Return existing Phase 1 movies
    SELECT array_agg(movie_id ORDER BY created_at)
    INTO v_all_movies
    FROM session_phase1_movies
    WHERE session_id = p_session_id;
    
    RETURN v_all_movies;
  END IF;
  
  -- Step 1: Select 10 random movies from top_vote_count = 1
  SELECT array_agg(id)
  INTO v_vote_count_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_count = 1
    ORDER BY random()
    LIMIT 10
  ) t;
  
  -- Insert vote_count selections
  IF v_vote_count_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_count_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'vote_count');
    END LOOP;
  END IF;
  
  -- Step 2: Select 10 random movies from top_vote_variance = 1 (excluding Step 1)
  SELECT array_agg(id)
  INTO v_vote_variance_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_variance = 1
      AND id != ALL(COALESCE(v_vote_count_movies, ARRAY[]::text[]))
    ORDER BY random()
    LIMIT 10
  ) t;
  
  -- Insert vote_variance selections
  IF v_vote_variance_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_variance_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'vote_variance');
    END LOOP;
  END IF;
  
  -- Step 3: Select 1 movie per genre (1-10)
  v_genre_movies := ARRAY[]::text[];
  FOR v_genre_num IN 1..10 LOOP
    SELECT id INTO v_movie_id
    FROM movies
    WHERE top_genre = v_genre_num
      AND id != ALL(COALESCE(v_vote_count_movies, ARRAY[]::text[]))
      AND id != ALL(COALESCE(v_vote_variance_movies, ARRAY[]::text[]))
    ORDER BY random()
    LIMIT 1;
    
    IF v_movie_id IS NOT NULL THEN
      v_genre_movies := array_append(v_genre_movies, v_movie_id);
      
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'genre_' || v_genre_num);
      
      v_movie_id := NULL; -- Reset for next iteration
    END IF;
  END LOOP;
  
  -- Combine all movies
  v_all_movies := COALESCE(v_vote_count_movies, ARRAY[]::text[]) || 
                  COALESCE(v_vote_variance_movies, ARRAY[]::text[]) || 
                  v_genre_movies;
  
  RETURN v_all_movies;
END;
$$;
