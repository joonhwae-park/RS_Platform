export interface Movie {
  id: string;
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
  movieId: string;
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
  serendipityAttitude: string;
  noveltyAttitude: string;
  diversityAttitude: string;
  noveltyAttitude2: string;
  diversityAttitude2: string;
  serendipityAttitude2: string;
  nationality?: string;
  occupation?: string;
  additionalComments?: string;
  email?: string;
}

export interface PhaseTransition {
  sessionId: string;
  fromPhase: string;
  toPhase: string;
  timestamp: string;
}

export interface TrailerView {
  sessionId: string;
  movieId: string;
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