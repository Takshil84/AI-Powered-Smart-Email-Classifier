import React, { useState, useEffect, useMemo } from 'react';
import { ThemeProvider, createTheme, Box, CssBaseline, Snackbar, Alert } from '@mui/material';
import Navbar from './components/Navbar';
import Analysis from './pages/Analysis';
import Analytics from './pages/Analytics';
import History from './pages/History';
import axios from 'axios';

// -- Theme Generator --
const getTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: { main: mode === 'dark' ? '#818cf8' : '#0f172a' },
    background: { 
      default: mode === 'dark' ? '#0f172a' : '#f1f5f9', 
      paper: mode === 'dark' ? '#1e293b' : '#ffffff' 
    },
    text: { 
      primary: mode === 'dark' ? '#f1f5f9' : '#1e293b', 
      secondary: mode === 'dark' ? '#94a3b8' : '#64748b' 
    },
    success: { main: '#10b981' },
    warning: { main: '#f59e0b' },
    error: { main: '#ef4444' },
    divider: mode === 'dark' ? '#334155' : '#e2e8f0',
  },
  typography: {
    fontFamily: '"Inter", "system-ui", sans-serif',
    h4: { fontWeight: 800, letterSpacing: '-0.02em' },
    h5: { fontWeight: 700, letterSpacing: '-0.01em' },
    h6: { fontWeight: 600 },
    subtitle1: { fontWeight: 600 },
    subtitle2: { fontWeight: 600 },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { borderRadius: 8, boxShadow: 'none' },
        contained: { '&:hover': { boxShadow: '0 4px 12px rgba(15, 23, 42, 0.15)' } }
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: mode === 'dark' ? '#1e293b' : '#ffffff',
          boxShadow: mode === 'dark' 
            ? '0 1px 3px 0 rgb(0 0 0 / 0.3)' 
            : '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          borderRadius: 12,
          border: `1px solid ${mode === 'dark' ? '#334155' : '#e2e8f0'}`,
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderColor: mode === 'dark' ? '#334155' : '#e2e8f0',
        },
        head: {
          backgroundColor: mode === 'dark' ? '#1e293b' : '#f8fafc',
          color: mode === 'dark' ? '#94a3b8' : '#64748b',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: mode === 'dark' ? '#0f172a' : '#f8fafc',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontWeight: 600, borderRadius: 6 },
      },
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState(0);
  
  // -- Dark Mode State --
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved !== null ? JSON.parse(saved) : true;
  });
  
  // -- Toast State --
  const [toast, setToast] = useState({ open: false, message: '', severity: 'success' });
  
  // -- Confidence Threshold State --
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  
  // -- Other State --
  const [subject, setSubject] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);

  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('email_focus_v1');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  // Persist dark mode
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('email_focus_v1', JSON.stringify(history));
  }, [history]);

  // Create theme based on mode
  const theme = useMemo(() => getTheme(darkMode ? 'dark' : 'light'), [darkMode]);

  // Toast helpers
  const showToast = (message, severity = 'success') => {
    setToast({ open: true, message, severity });
  };
  
  const hideToast = () => {
    setToast(prev => ({ ...prev, open: false }));
  };

  // -- Classification Logic --
  const handleClassify = async () => {
    if (!content.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const API_URL = import.meta.env.VITE_API_URL || '';
      const response = await axios.post(`${API_URL}/api/classify`, { subject, content });
      const result = {
        id: Date.now(),
        subject,
        text: content,
        ...response.data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setLastResult(result);
      setHistory(prev => [result, ...prev]);
      
      // Show toast based on result
      if (result.confidence < confidenceThreshold) {
        showToast(`Low confidence result (${(result.confidence * 100).toFixed(0)}%)`, 'warning');
      } else if (result.urgency === 'High') {
        showToast('⚠️ High urgency email detected!', 'error');
      } else {
        showToast('Email analyzed successfully', 'success');
      }
    } catch (err) {
      setError("System Unreachable. Is backend running?");
      showToast('Analysis failed. Check backend connection.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    showToast('History cleared', 'info');
  };
  
  const deleteHistoryItem = (id) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const getUrgencyColor = (u) => {
    switch((u || "").toLowerCase()) {
      case 'high': return theme.palette.error.main;
      case 'medium': return theme.palette.warning.main;
      case 'low': return theme.palette.success.main;
      default: return theme.palette.text.secondary;
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', pb: 8, transition: 'background-color 0.3s ease' }}>
        
        <Navbar 
            currentTab={activeTab} 
            onTabChange={handleTabChange}
            stats={{
                total: history.length,
                urgent: history.filter(h => (h.urgency || "").toLowerCase() === 'high').length
            }}
            darkMode={darkMode}
            onToggleDarkMode={toggleDarkMode}
        />

        {activeTab === 0 && (
            <Analysis 
                subject={subject} setSubject={setSubject}
                content={content} setContent={setContent}
                loading={loading} error={error}
                handleClassify={handleClassify}
                lastResult={lastResult}
                setLastResult={setLastResult}
                getUrgencyColor={getUrgencyColor}
                confidenceThreshold={confidenceThreshold}
            />
        )}
        
        {activeTab === 1 && (
            <Analytics history={history} />
        )}
        
        {activeTab === 2 && (
            <History 
                history={history} 
                clearHistory={clearHistory} 
                deleteHistoryItem={deleteHistoryItem}
                getUrgencyColor={getUrgencyColor} 
            />
        )}

        {/* Toast Notification */}
        <Snackbar 
          open={toast.open} 
          autoHideDuration={4000} 
          onClose={hideToast}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={hideToast} severity={toast.severity} variant="filled" sx={{ width: '100%' }}>
            {toast.message}
          </Alert>
        </Snackbar>

      </Box>
    </ThemeProvider>
  );
}

export default App;
