import React from 'react';

interface ProgressBarProps {
  current: number;
  total: number;
  minimum?: number;
  label: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ 
  current, 
  total, 
  minimum, 
  label 
}) => {
  const percentage = (current / total) * 100;
  const minimumPercentage = minimum ? (minimum / total) * 100 : 0;

  return (
    <div className="w-full mb-8">
      <div className="flex justify-between items-center mb-2">
        <div className="flex flex-col gap-1">
          <span className="text-amber-400 font-medium">{label}</span>
          {label.includes('Initial') && (
            <span className="text-gray-400 text-sm italic">
              💡 Rate more movies to get more accurate recommendations!
            </span>
          )}
        </div>
        <span className="text-white">
          {current}/{total}
          {minimum && current < minimum && (
            <span className="text-amber-500 ml-2">(minimum {minimum})</span>
          )}
        </span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden relative">
        <div
          className="h-full bg-gradient-to-r from-amber-500 to-amber-600 transition-all duration-500 ease-out"
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
        {minimum && (
          <div
            className="absolute top-0 w-0.5 h-full bg-red-500 z-10"
            style={{ left: `${minimumPercentage}%` }}
          />
        )}
      </div>
    </div>
  );
};