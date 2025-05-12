import React from 'react'
import { DollarSign, TrendingUp } from 'lucide-react'

function PredictionResult({ prediction }) {
  if (!prediction) return null
  
  return (
    <div className="mt-8 animate-fadeIn">
      <div className="divider"></div>
      <h3 className="text-xl font-bold mb-4 text-gray-800 flex items-center gap-2">
        <TrendingUp className="h-5 w-5 text-green-500" />
        Prediction Result
      </h3>
      
      <div className="stats shadow w-full bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100">
        <div className="stat">
          <div className="stat-figure text-blue-500">
            <DollarSign className="h-8 w-8" />
          </div>
          <div className="stat-title text-gray-600">Estimated House Price</div>
          <div className="stat-value text-blue-600">{prediction.formattedPrice}</div>
        </div>
      </div>
      
      <div className="alert bg-blue-50 shadow-sm mt-4 border-l-4 border-blue-500">
        <div>
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="stroke-blue-500 flex-shrink-0 w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span className="text-gray-700">This prediction is based on the features you provided and our MLOps model.</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PredictionResult