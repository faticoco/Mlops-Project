import React from 'react'
import HousePriceForm from './components/HousePriceForm'
import Header from './components/Header'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <HousePriceForm />
      </main>
      <footer className="py-4 text-center text-gray-500 text-sm">
        © 2025 MLOps House Price Prediction System
      </footer>
    </div>
  )
}

export default App