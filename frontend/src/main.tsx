import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const application = import.meta.env.MODE === 'showcase' ? import('./ShowcaseApp') : import('./App')
application.then(({ default: App }) => createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>))
