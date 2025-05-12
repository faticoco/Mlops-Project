import React from 'react'
import { Home } from 'lucide-react'

function Header() {
  return (
    <div className="navbar bg-white shadow-md">
      <div className="container mx-auto">
        <div className="flex-1">
          <a className="flex items-center gap-2 text-xl font-semibold text-gray-800">
            <Home className="h-6 w-6 text-blue-500" />
            <span>House Price Prediction</span>
          </a>
        </div>
        <div className="flex-none">
          <span className="text-sm text-gray-600">MLOps Project</span>
        </div>
      </div>
    </div>
  )
}

export default Header