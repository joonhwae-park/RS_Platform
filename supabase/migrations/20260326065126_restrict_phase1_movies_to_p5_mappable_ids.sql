/*
  # Restrict Phase 1 movie selection to P5-mappable IDs

  ## Root Cause
  The P5 recommendation model was trained on a dataset whose datamaps.json only
  contains older IMDB-style movie IDs (7-digit IDs with a leading zero, e.g. "0120815").
  Newer movies have IDs without a leading zero (e.g. "1663202").

  After the vote_count-weighted sampling was introduced, Phase 1 began preferentially
  surfacing newer, higher-vote-count movies (which tend to have non-leading-zero IDs).
  Users rate these newer movies, but map_history_for_p5() cannot map them to internal
  P5 item IDs. This leaves P5 with an empty history context, causing it to generate
  unparseable output and record a score of -1 for every recommendation.

  ## Fix
  Add `AND id LIKE '0%'` to all three selection steps so that Phase 1 only presents
  movies whose IDs begin with '0' — the format guaranteed to exist in P5's datamaps.

  ## Impact
  - top_vote_count pool: 82 → 25 movies (still sufficient for 10 per batch)
  - top_vote_variance pool: 76 → 24 movies (still sufficient for 10 per batch)
  - genre pool: 221 → 263 old-format movies available across genres
*/

CREATE OR REPLACE FUNCTION select_phase1_movies_for_session(
  p_session_id uuid,
  p_batch_number integer DEFAULT 1
)
RETURNS text[]
LANGUAGE plpgsql
AS $$
DECLARE
  v_existing_batch text[];
  v_vote_count_movies text[];
  v_vote_variance_movies text[];
  v_genre_movies text[];
  v_all_movies text[];
  v_movie_id text;
  v_genre_num integer;
  v_excluded_movies text[];
  v_total_loaded integer;
  v_remaining integer;
  v_limit_per_step integer;
  v_genre_limit integer;
  v_batch_limit integer;
BEGIN
  IF p_batch_number < 1 OR p_batch_number > 4 THEN
    RETURN ARRAY[]::text[];
  END IF;

  SELECT array_agg(movie_id ORDER BY created_at)
  INTO v_existing_batch
  FROM session_phase1_movies
  WHERE session_id = p_session_id
    AND selection_type LIKE 'batch_' || p_batch_number || '_%';

  IF v_existing_batch IS NOT NULL AND array_length(v_existing_batch, 1) > 0 THEN
    RETURN v_existing_batch;
  END IF;

  SELECT COALESCE(array_agg(movie_id), ARRAY[]::text[])
  INTO v_excluded_movies
  FROM session_phase1_movies
  WHERE session_id = p_session_id;

  SELECT COUNT(*) INTO v_total_loaded
  FROM session_phase1_movies
  WHERE session_id = p_session_id;

  v_remaining := 100 - v_total_loaded;

  IF v_remaining <= 0 THEN
    RETURN ARRAY[]::text[];
  END IF;

  v_batch_limit := LEAST(30, v_remaining);
  v_limit_per_step := v_batch_limit / 3;
  v_genre_limit := v_batch_limit - (v_limit_per_step * 2);

  -- Step 1: Select from top_vote_count, restricted to old-format IDs (P5-mappable)
  SELECT array_agg(id)
  INTO v_vote_count_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_count = 1
      AND id LIKE '0%'
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT v_limit_per_step
  ) t;

  IF v_vote_count_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_count_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_vote_count');
    END LOOP;
  END IF;

  v_excluded_movies := v_excluded_movies || COALESCE(v_vote_count_movies, ARRAY[]::text[]);

  -- Step 2: Select from top_vote_variance, restricted to old-format IDs (P5-mappable)
  SELECT array_agg(id)
  INTO v_vote_variance_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_variance = 1
      AND id LIKE '0%'
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT v_limit_per_step
  ) t;

  IF v_vote_variance_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_variance_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_vote_variance');
    END LOOP;
  END IF;

  v_excluded_movies := v_excluded_movies || COALESCE(v_vote_variance_movies, ARRAY[]::text[]);

  -- Step 3: Select per genre, restricted to old-format IDs (P5-mappable)
  v_genre_movies := ARRAY[]::text[];
  FOR v_genre_num IN 1..10 LOOP
    EXIT WHEN array_length(COALESCE(v_genre_movies, ARRAY[]::text[]), 1) >= v_genre_limit;

    SELECT id INTO v_movie_id
    FROM movies
    WHERE top_genre = v_genre_num
      AND id LIKE '0%'
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT 1;

    IF v_movie_id IS NOT NULL THEN
      v_genre_movies := array_append(v_genre_movies, v_movie_id);

      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_genre_' || v_genre_num);

      v_excluded_movies := v_excluded_movies || ARRAY[v_movie_id];
      v_movie_id := NULL;
    END IF;
  END LOOP;

  v_all_movies := COALESCE(v_vote_count_movies, ARRAY[]::text[]) ||
                  COALESCE(v_vote_variance_movies, ARRAY[]::text[]) ||
                  v_genre_movies;

  RETURN v_all_movies;
END;
$$;
