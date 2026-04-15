export interface AnalysisResult {
	quality: 'good' | 'bad' | 'error';
	verdict_ru: string;
	confidence: 'high' | 'medium' | 'low';
	defects_found: string[];
	reasoning: string;
	no_egg?: boolean;
}

export interface ProgressLog {
	text: string;
	kind: 'stage' | 'progress';
	stage?: number;
}
