import React, { useState } from 'react'
import LandscapeView from './LandscapeView.jsx'
import Harmonize from './scene/Harmonize.jsx'

// Two-act story: Act 1 harmonizes messy datasets (progressive PCA + ComBat), Act 2
// drapes validated risk over the harmonized map. The switcher persists the choice
// so a reload keeps you where you were.
const VIEW_KEY = 'geo.view'

export default function App() {
  const [view, setView] = useState(() => {
    try {
      return localStorage.getItem(VIEW_KEY) || 'harmonize'
    } catch {
      return 'harmonize'
    }
  })

  const choose = (v) => {
    setView(v)
    try {
      localStorage.setItem(VIEW_KEY, v)
    } catch {
      /* ignore storage errors (private mode) */
    }
  }

  return (
    <div className="app">
      <div className="viewswitch">
        <button className={view === 'harmonize' ? 'on' : ''} onClick={() => choose('harmonize')}>
          ① Harmonize
        </button>
        <button className={view === 'landscape' ? 'on' : ''} onClick={() => choose('landscape')}>
          ② Landscape
        </button>
      </div>

      {view === 'harmonize' ? <Harmonize /> : <LandscapeView />}
    </div>
  )
}
