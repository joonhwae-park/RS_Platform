import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  db: {
    schema: 'public',
  },
  auth: {
    autoRefreshToken: true,
    persistSession: false,
  },
  global: {
    headers: {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0'
    }
  }
});

export type Database = {
  public: {
    Tables: {
      user_sessions: {
        Row: {
          id: string;
          user_id: string;
          phase: string;
          start_time: string;
          end_time: string | null;
          created_at: string;
          minimum_ratings_required: number;
          movie_genre_preferences: string[];
          openness_to_experience: string;
          risk_aversion: string;
          movie_expertise: string;
          attention_check: string;
          gender?: string;
          ageRange?: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          phase: string;
          start_time: string;
          end_time?: string | null;
          created_at?: string;
          minimum_ratings_required?: number;
          movie_genre_preferences: string[];
          openness_to_experience?: string;
          risk_aversion?: string;
          movie_expertise?: string;
          attention_check?: string;
          gender?: string;
          ageRange?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          phase?: string;
          start_time?: string;
          end_time?: string | null;
          created_at?: string;
          minimum_ratings_required?: number;
          movie_genre_preferences?: string[];
          openness_to_experience?: string;
          risk_aversion?: string;
          movie_expertise?: string;
          attention_check?: string;
          gender?: string;
          ageRange?: string;
        };
      };
      movie_ratings: {
        Row: {
          id: string;
          session_id: string;
          movie_id: number;
          rating: number;
          diversity_rating: number | null;
          novelty_rating: number | null;
          serendipity_rating: number | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          session_id: string;
          movie_id: number;
          rating: number;
          diversity_rating?: number | null;
          novelty_rating?: number | null;
          serendipity_rating?: number | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          session_id?: string;
          movie_id?: number;
          rating?: number;
          diversity_rating?: number | null;
          novelty_rating?: number | null;
          serendipity_rating?: number | null;
          created_at?: string;
        };
      };
      mouse_events: {
        Row: {
          id: string;
          session_id: string;
          x: number;
          y: number;
          timestamp: string;
          created_at: string;
        };
        Insert: {
          id?: string;
          session_id: string;
          x: number;
          y: number;
          timestamp: string;
          created_at?: string;
        };
        Update: {
          id?: string;
          session_id?: string;
          x?: number;
          y?: number;
          timestamp?: string;
          created_at?: string;
        };
      };
      movies: {
        Row: {
          id: number;
          title: string;
          description: string;
          poster: string;
          trailer: string;
          genre: string;
          year: number;
          director: string;
          actors: string;
          created_at: string;
        };
        Insert: {
          id: number;
          title: string;
          description: string;
          poster: string;
          trailer: string;
          genre: string;
          year: number;
          director: string;
          actors: string;
          created_at?: string;
        };
        Update: {
          id?: number;
          title?: string;
          description?: string;
          poster?: string;
          trailer?: string;
          genre?: string;
          year?: number;
          director?: string;
          actors?: string;
          created_at?: string;
        };
      };
    };
  };
};