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
  private config: RecommenderConfig = {
    ratingWeight: 0.4,
    genreWeight: 0.3,
    yearWeight: 0.1,
    directorWeight: 0.1,
    diversityBoost: 0.1
  };

  private userFactors: SVDUserFactors | null = null;
  private itemFactors: SVDItemFactors | null = null;
  private factorsLoaded: boolean = false;

  /**
   * Load SVD factors from Supabase storage
   */
  private async loadSVDFactors(): Promise<void> {
    if (this.factorsLoaded) return;

    try {
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      
      console.log('Loading SVD factors from:', supabaseUrl);
      
      // Load user factors
      const userFactorsResponse = await fetch(
        `${supabaseUrl}/storage/v1/object/public/models/svd_user_factors.json`
      );
      
      if (!userFactorsResponse.ok) {
        console.warn(`Failed to load user factors: ${userFactorsResponse.status}`);
        this.userFactors = null;
        this.itemFactors = null;
        this.factorsLoaded = false;
        return;
      }
      
      this.userFactors = await userFactorsResponse.json();
      
      // Load item factors
      const itemFactorsResponse = await fetch(
        `${supabaseUrl}/storage/v1/object/public/models/svd_item_factors.json`
      );
      
      if (!itemFactorsResponse.ok) {
        console.warn(`Failed to load item factors: ${itemFactorsResponse.status}`);
        this.userFactors = null;
        this.itemFactors = null;
        this.factorsLoaded = false;
        return;
      }
      
      this.itemFactors = await itemFactorsResponse.json();
      
      this.factorsLoaded = true;
      console.log('SVD factors loaded successfully');
      console.log('User factors count:', Object.keys(this.userFactors || {}).length);
      console.log('Item factors count:', Object.keys(this.itemFactors || {}).length);
      
    } catch (error) {
      console.error('Error loading SVD factors:', error);
      this.userFactors = null;
      this.itemFactors = null;
      this.factorsLoaded = false;
    }
  }

  /**
   * Calculate dot product between two vectors
   */
  private dotProduct(vectorA: number[], vectorB: number[]): number {
    if (vectorA.length !== vectorB.length) {
      throw new Error('Vectors must have the same length');
    }
    
    return vectorA.reduce((sum, a, i) => sum + a * vectorB[i], 0);
  }

  /**
   * Create a synthetic user profile based on Phase 1 ratings
   */
  private createSyntheticUserProfile(userRatings: any[]): number[] | null {
    if (!this.itemFactors || !userRatings.length) return null;

    try {
      // Get the dimensionality from the first item factor
      const firstItemId = Object.keys(this.itemFactors)[0];
      const dimensions = this.itemFactors[firstItemId]?.length || 50;
      
      // Initialize user profile with zeros
      const userProfile = new Array(dimensions).fill(0);
      let totalWeight = 0;

      // Aggregate item factors weighted by user ratings
      for (const rating of userRatings) {
        const movieId = String(rating.movieId);
        const itemFactor = this.itemFactors[movieId];
        
        if (itemFactor && rating.rating > 0) {
          // Normalize rating to [-1, 1] range (assuming ratings are 0-10)
          const normalizedRating = (rating.rating - 5) / 5;
          
          // Add weighted item factors to user profile
          for (let i = 0; i < dimensions; i++) {
            userProfile[i] += itemFactor[i] * normalizedRating;
          }
          totalWeight += Math.abs(normalizedRating);
        }
      }

      // Normalize the user profile
      if (totalWeight > 0) {
        for (let i = 0; i < dimensions; i++) {
          userProfile[i] /= totalWeight;
        }
      }

      return userProfile;
    } catch (error) {
      console.error('Error creating synthetic user profile:', error);
      return null;
    }
  }

  /**
   * Main recommendation algorithm using SVD
   */
  async generateRecommendations(
    sessionId: string,
    userRatings: any[]
  ): Promise<number[]> {
    try {
      console.log('Generating recommendations for session:', sessionId);
      console.log('User ratings count:', userRatings.length);
      
      // Load SVD factors if not already loaded
      await this.loadSVDFactors();

      if (!this.itemFactors) {
        console.warn('SVD factors not available, falling back to simple recommender');
        return this.getFallbackRecommendations();
      }

      // Filter valid ratings (rating > 0)
      const validRatings = userRatings.filter(r => r.rating > 0);
      console.log('Valid ratings count:', validRatings.length);
      
      if (validRatings.length === 0) {
        console.warn('No valid ratings found, using fallback recommendations');
        return this.getFallbackRecommendations();
      }

      // Create synthetic user profile from Phase 1 ratings
      const userProfile = this.createSyntheticUserProfile(validRatings);
      
      if (!userProfile) {
        console.warn('Could not create user profile, using fallback recommendations');
        return this.getFallbackRecommendations();
      }

      console.log('User profile created with dimensions:', userProfile.length);

      // Get all Phase 2 movies
      const { data: phase2Movies, error } = await supabase
        .from('phase2_movies')
        .select('id');

      if (error) throw error;

      if (!phase2Movies || phase2Movies.length === 0) {
        console.warn('No Phase 2 movies found');
        return [];
      }

      console.log('Phase 2 movies count:', phase2Movies.length);

      // Get movies user hasn't rated yet
      const ratedMovieIds = userRatings.map(r => String(r.movieId));
      const phase2MovieIds = phase2Movies.map(m => String(m.id));
      const candidateMovieIds = phase2MovieIds.filter(id => !ratedMovieIds.includes(id));

      console.log('Candidate movies for recommendation:', candidateMovieIds.length);

      // Calculate SVD scores for candidate movies
      const recommendations: { movieId: number; score: number }[] = [];

      for (const movieId of candidateMovieIds) {
        const itemFactor = this.itemFactors[movieId];
        
        if (itemFactor) {
          try {
            // Calculate SVD prediction score
            const svdScore = this.dotProduct(userProfile, itemFactor);
            
            // Add some diversity and randomness
            const diversityBoost = Math.random() * 0.1;
            const finalScore = svdScore + diversityBoost;
            
            recommendations.push({
              movieId: parseInt(movieId),
              score: finalScore
            });
          } catch (error) {
            console.warn(`Error calculating score for movie ${movieId}:`, error);
          }
        } else {
          console.warn(`No item factor found for movie ${movieId}`);
        }
      }

      console.log('Recommendations calculated:', recommendations.length);

      // Sort by score (descending) and return top 10
      recommendations.sort((a, b) => b.score - a.score);
      const topRecommendations = recommendations.slice(0, 10).map(r => r.movieId);

      console.log('SVD recommendations generated:', topRecommendations);
      console.log('Recommendation scores:', recommendations.slice(0, 10).map(r => ({ id: r.movieId, score: r.score.toFixed(4) })));

      return topRecommendations;

    } catch (error) {
      console.error('Error in SVD recommendation generation:', error);
      return this.getFallbackRecommendations();
    }
  }

  /**
   * Analyze user's rating patterns to create a user profile
   */
  private analyzeUserProfile(ratings: any[]) {
    const validRatings = ratings.filter(r => r.rating > 0);
    
    if (validRatings.length === 0) {
      return {
        averageRating: 5,
        preferredGenres: [],
        preferredYears: [],
        preferredDirectors: []
      };
    }

    const averageRating = validRatings.reduce((sum, r) => sum + r.rating, 0) / validRatings.length;
    
    return {
      averageRating,
      preferredGenres: [],
      preferredYears: [],
      preferredDirectors: []
    };
  }

  /**
   * Calculate recommendation score for a movie based on user profile
   * This is the fallback algorithm when SVD is not available
   */
  private calculateRecommendationScore(movie: any, userProfile: any, allMovies: any[]): number {
    let score = 0;

    // Base score from movie's general appeal
    score += movie.year > 2010 ? 0.2 : 0.1;
    
    // Diversity boost
    score += this.calculateDiversityBoost(movie, allMovies) * this.config.diversityBoost;
    
    // Add randomness for serendipity
    score += Math.random() * 0.1;

    return score;
  }

  /**
   * Calculate diversity boost for a movie
   */
  private calculateDiversityBoost(movie: any, allMovies: any[]): number {
    const genreCount = allMovies.filter(m => m.genre === movie.genre).length;
    const totalMovies = allMovies.length;
    
    return 1 - (genreCount / totalMovies);
  }

  /**
   * Fallback recommendations when SVD algorithm fails
   */
  private async getFallbackRecommendations(): Promise<number[]> {
    try {
      console.log('Using fallback recommendation algorithm');
      
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

  /**
   * Update configuration for A/B testing different algorithms
   */
  updateConfig(newConfig: Partial<RecommenderConfig>) {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Log recommendation results for analysis
   */
  async logRecommendation(sessionId: string, recommendedMovieIds: number[]) {
    try {
      console.log('SVD recommendations for session', sessionId, ':', recommendedMovieIds);
      
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

// Import supabase here to avoid circular dependency
import { supabase } from '../lib/supabase';

export const recommenderService = new RecommenderService();