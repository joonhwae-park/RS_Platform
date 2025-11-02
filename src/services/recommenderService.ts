interface SVDUserFactors {
  [userId: string]: number[];
}

interface SVDItemFactors {
  [movieId: string]: number[];
}

export interface RecommenderConfig {
  // Weight factors for different aspects of recommendation
  ratingWeight: number;
  genreWeight: number;
  yearWeight: number;
  directorWeight: number;
  diversityBoost: number;
}

export class RecommenderService {
  /**
   * Trigger recommendation generation via the Cloud Run API
   */
  async triggerRecommendationGeneration(sessionId: string): Promise<boolean> {
    try {
      console.log('Triggering recommendation generation for session:', sessionId);
      console.log('Environment check - API URL configured:', !!import.meta.env.VITE_RECOMMENDATION_API_URL);
      
      const result = await recommendationAPI.generateRecommendations(sessionId);
      
      if (result) {
        console.log('Recommendations generated successfully:', result);
        return true;
      } else {
        console.warn('Failed to generate recommendations via API');
        return false;
      }
    } catch (error) {
      console.error('Error triggering recommendation generation:', error);
      return false;
    }
  }

  /**
   * Get recommendations from the recommendations table in Supabase
   */
  async generateRecommendations(
    sessionId: string,
    userRatings: any[]
  ): Promise<number[]> {
    try {
      console.log('Reading recommendations from database for session:', sessionId);
      
      // Get recommendations from the recommendations table, sorted by display_order
      // Use the latest recommendations only (by created_at)
      const { data: allRecs, error: fetchError } = await supabase
        .from('recommendations')
        .select('movie_id, display_order, created_at')
        .eq('session_id', sessionId)
        .not('display_order', 'is', null)
        .order('created_at', { ascending: false });

      if (fetchError) {
        console.error('Error fetching recommendations from database:', fetchError);
        return this.getFallbackRecommendations();
      }

      if (!allRecs || allRecs.length === 0) {
        console.warn('No recommendations found in database for session:', sessionId);
        return this.getFallbackRecommendations();
      }

      // Get the most recent timestamp
      const latestTimestamp = allRecs[0].created_at;

      // Filter to only include recommendations from the latest batch
      const latestRecs = allRecs.filter(r => r.created_at === latestTimestamp);

      // Sort by display_order
      const recommendations = latestRecs.sort((a, b) => (a.display_order || 0) - (b.display_order || 0));

      // Handle potential duplicate movie_ids by taking only unique ones
      const uniqueMovieIds = Array.from(new Set(recommendations.map(r => r.movie_id)));

      console.log('Recommendations loaded from database:', uniqueMovieIds);
      console.log('Latest timestamp:', latestTimestamp);
      console.log('Total recommendations found:', allRecs.length);
      console.log('Latest batch recommendations:', recommendations.length);

      return uniqueMovieIds;

    } catch (error) {
      console.error('Error reading recommendations from database:', error);
      return this.getFallbackRecommendations();
    }
  }

  /**
   * Fallback recommendations when SVD algorithm fails
   */
  private async getFallbackRecommendations(): Promise<number[]> {
    try {
      console.log('Using fallback: getting first 10 phase2_movies');
      
      const { data: phase2Movies, error } = await supabase
        .from('phase2_movies')
        .select('id')
        .limit(10);

      if (error) throw error;
      
      const movieIds = phase2Movies?.map(m => m.id) || [];
      
      // Shuffle the array for some randomness
      for (let i = movieIds.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [movieIds[i], movieIds[j]] = [movieIds[j], movieIds[i]];
      }
      
      console.log('Using fallback recommendations:', movieIds);
      return movieIds;
    } catch (error) {
      console.error('Error getting fallback recommendations:', error);
      return [];
    }
  }

  async checkHealth(): Promise<boolean> {
    return await recommendationAPI.checkHealth();
  }
  async logRecommendation(sessionId: string, recommendedMovieIds: number[]) {
    try {
      console.log('Recommendations for session', sessionId, ':', recommendedMovieIds);
      
      // Optional: Store in database for later analysis
      // await supabase.from('recommendations_log').insert({
      //   session_id: sessionId,
      //   recommended_movies: recommendedMovieIds,
      //   algorithm_version: 'SVD-1.0',
      //   created_at: new Date().toISOString()
      // });
    } catch (error) {
      console.error('Error logging recommendation:', error);
    }
  }
}

import { recommendationAPI } from './recommendationAPI';
import { supabase } from '../lib/supabase';
export const recommenderService = new RecommenderService();