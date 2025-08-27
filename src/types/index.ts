export interface Movie {
  id: number;
  title: string;
  description: string;
  poster: string;
  trailer: string;
  genre: string;
  year: number;
  director: string;
  actors: string;
}

export interface Rating {
  movieId: number;
  rating: number;
  diversity?: number;
  novelty?: number;
  serendipity?: number;
}

export interface QuestionnaireData {
  movieWatchingFrequency: string;
  streamingServices: string[];
  primaryStreamingService: string;
  movieGenrePreferences: string[];
  opennessToExperience: string;
  riskAversion: string;
  movieExpertise: string;
  gender?: string;
  ageRange?: string;
  attentionCheck: string;
  nationality?: string;
  occupation?: string;
  additionalComments?: string;
}

export interface PhaseTransition {
  sessionId: string;
  fromPhase: string;
  toPhase: string;
  timestamp: string;
}

export interface TrailerView {
  sessionId: string;
  movieId: number;
  startTime: string;
  endTime?: string;
  duration?: number;
}

export interface SessionData {
  userId: string;
  phase: 'initial' | 'recommendation';
  startTime: number;
  endTime?: number;
  ratings: Rating[];
  minimumRatingsRequired: number;
  screenWidth?: number;
  screenHeight?: number;
}

export type AppPhase = 'intro' | 'initial' | 'choice' | 'recommendation' | 'questionnaire' | 'complete';