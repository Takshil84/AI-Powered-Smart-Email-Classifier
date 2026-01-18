import React from 'react';
import { 
  Box, Container, Grid, Paper, Typography, TextField, Button, 
  Chip, Divider, LinearProgress, IconButton, Tooltip, Alert
} from '@mui/material';
import { 
    Send as SendIcon, 
    AutoAwesome as AutoAwesomeIcon,
    Subject as SubjectIcon,
    Segment as ContentIcon,
    DeleteOutline as ClearIcon,
    Lightbulb as IdeaIcon,
    Bolt as ActionIcon,
    ContentPaste as PasteIcon,
    Download as DownloadIcon
} from '@mui/icons-material';

import { renderHeatmap } from '../utils/heatmap';
import { exportAnalysisPDF } from '../utils/pdfExport';

const SAMPLE_EMAIL = {
    subject: "Urgent: System Failure in Production",
    content: "Hi Support Team, \n\nThe production database is unresponsive and customers are unable to checkout. This is starting to affect our revenue significantly. We need immediate assistance to resolve this critical issue.\n\nPlease acknowledge receipt.\n\nRegards,\nIT Ops"
};

function Analysis({ 
    subject, setSubject, 
    content, setContent, 
    loading, error, handleClassify, 
    lastResult, setLastResult, getUrgencyColor,
    confidenceThreshold 
}) {

  const isLowConfidence = lastResult && lastResult.confidence < confidenceThreshold;

  const handleSample = () => {
    setSubject(SAMPLE_EMAIL.subject);
    setContent(SAMPLE_EMAIL.content);
  };

  const handleClear = () => {
    setSubject("");
    setContent("");
    setLastResult(null);
  };



  return (
    <Container 
      maxWidth={false}  
      disableGutters
      sx={{ 
        mt: { xs: 2, sm: 4 }, 
        height: { xs: 'auto', md: 'calc(100vh - 120px)' }, 
        pb: 2,
        px: { xs: 1, sm: 2, md: 3 },
        width: '100%'
      }}
    >
        <Grid 
          container 
          spacing={2} 
          sx={{ 
            height: '100%',
            width: '100%',
            margin: 0,
            flexDirection: { xs: 'column', md: 'row' },
            flexWrap: 'nowrap'
          }}
        >
            
            {/* LEFT: INPUT CARD */}
            <Grid 
              item 
              sx={{ 
                width: { xs: '100%', md: '50%' },
                height: { xs: 'auto', md: '100%' }, 
                display: 'flex', 
                flexDirection: 'column',
                minHeight: { xs: '450px', sm: '500px', md: 'auto' },
                flex: { xs: '0 0 auto', md: '1 1 50%' }
              }}
            >
                <Paper 
                    sx={{ 
                        flexGrow: 1, display: 'flex', flexDirection: 'column', 
                        overflow: 'hidden', 
                        width: '100%'
                    }} 
                >
                    {/* Header */}
                    <Box sx={{ 
                      p: { xs: 1, sm: 1.5 }, 
                      borderBottom: '1px solid #e2e8f0', 
                      display: 'flex', 
                      flexDirection: { xs: 'column', sm: 'row' },
                      justifyContent: 'space-between', 
                      alignItems: { xs: 'flex-start', sm: 'center' },
                      gap: { xs: 1, sm: 0 }
                    }}>
                        <Box>
                            <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }} color="text.primary">Input Email</Typography>
                            <Typography variant="body2" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }} color="text.secondary">Paste email details to classify.</Typography>
                        </Box>
                        <Button 
                            startIcon={<PasteIcon />} 
                            size="small" 
                            variant="outlined" 
                            onClick={handleSample} 
                            sx={{ 
                              borderColor: '#e2e8f0', 
                              color:'text.secondary',
                              fontSize: { xs: '0.75rem', sm: '0.875rem' }
                            }}
                        >
                            Use Sample
                        </Button>
                    </Box>

                    {/* Inputs */}
                    <Box sx={{ p: { xs: 1.5, sm: 2 }, flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 2, overflowY: 'auto' }}>
                        <Box>
                            <Typography variant="subtitle2" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }} gutterBottom>SUBJECT</Typography>
                            <TextField 
                                fullWidth placeholder="Enter subject line..."
                                value={subject} onChange={e => setSubject(e.target.value)}
                                sx={{ '& .MuiOutlinedInput-root': { bgcolor: 'background.default' } }}
                                InputProps={{ sx: { fontSize: { xs: '0.8rem', sm: '0.9rem' } } }}
                            />
                        </Box>
                        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                            <Typography variant="subtitle2" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }} gutterBottom>CONTENT</Typography>
                            <TextField 
                                fullWidth multiline placeholder="Paste email body..."
                                value={content} onChange={e => setContent(e.target.value)}
                                sx={{ 
                                    flexGrow: 1, 
                                    '& .MuiOutlinedInput-root': { bgcolor: 'background.default', height: '100%', alignItems: 'flex-start' },
                                    '& .MuiInputBase-input': { height: '100% !important', overflow: 'auto !important' } 
                                }}
                                InputProps={{ sx: { fontSize: { xs: '0.8rem', sm: '0.875rem' }, lineHeight: 1.6 } }}
                            />
                        </Box>
                    </Box>

                    {/* Footer */}
                    <Box sx={{ 
                      p: 1.5, 
                      borderTop: '1px solid',
                      borderColor: 'divider',
                      bgcolor: 'action.hover', 
                      display: 'flex', 
                      flexDirection: { xs: 'column', sm: 'row' },
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      gap: { xs: 1, sm: 0 }
                    }}>
                        <Button 
                          startIcon={<ClearIcon />} 
                          color="inherit" 
                          onClick={handleClear} 
                          disabled={!subject && !content}
                          size="small"
                          sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
                        >
                            Clear
                        </Button>
                         <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: { xs: '100%', sm: 'auto' } }}>
                            {loading && <Typography variant="body1" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, display: { xs: 'none', sm: 'block' } }} color="text.secondary">Analyzing...</Typography>}
                            <Button 
                                variant="contained" 
                                size="large" 
                                disableElevation
                                endIcon={!loading && <SendIcon />} 
                                onClick={handleClassify} 
                                disabled={loading || !content}
                                fullWidth={window.innerWidth < 600}
                                sx={{ 
                                  px: { xs: 2, sm: 4 },
                                  fontSize: { xs: '0.8rem', sm: '0.9rem' }
                                }}
                            >
                                {loading ? <LinearProgress sx={{ width: 100 }} /> : "Analyze Email"}
                            </Button>
                        </Box>
                    </Box>
                </Paper>
            </Grid>

            {/* RIGHT: RESULT CARD */}
            <Grid 
              item 
              sx={{ 
                width: { xs: '100%', md: '50%' },
                height: { xs: 'auto', md: '100%' }, 
                display: 'flex', 
                flexDirection: 'column',
                minHeight: { xs: '450px', sm: '500px', md: 'auto' },
                flex: { xs: '0 0 auto', md: '1 1 50%' }
              }}
            >
                <Paper 
                    sx={{ 
                        flexGrow: 1, display: 'flex', flexDirection: 'column', 
                        overflow: 'hidden', position: 'relative',
                        width: '100%'
                    }} 
                >
                    {!lastResult ? (
                        /* Empty State */
                        <Box sx={{ 
                          flexGrow: 1, 
                          display: 'flex', 
                          flexDirection: 'column', 
                          alignItems: 'center', 
                          justifyContent: 'center', 
                          p: { xs: 2, sm: 4 }, 
                          height: '100%', 
                          color: 'text.secondary' 
                        }}>
                            <Box sx={{ p: { xs: 3, sm: 4 }, bgcolor: 'action.hover', borderRadius: '50%', mb: 3 }}>
                                <IdeaIcon sx={{ fontSize: { xs: 32, sm: 40 }, color: 'text.disabled' }} />
                            </Box>
                            <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }} color="text.primary" gutterBottom>AI Interpretation</Typography>
                            <Typography variant="body2" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' }, maxWidth: 300 }} align="center">
                                Waiting for input. The analysis will appear here with urgency scoring and recommended actions.
                            </Typography>
                        </Box>
                    ) : (
                        /* Result State */
                        <>
                             {/* Header */}
                            <Box sx={{ 
                              p: { xs: 1, sm: 1.5 }, 
                              borderBottom: '1px solid',
                              borderColor: 'divider',
                              display: 'flex', 
                              flexDirection: { xs: 'column', sm: 'row' },
                              alignItems: { xs: 'flex-start', sm: 'center' }, 
                              justifyContent: 'space-between', 
                              width: '100%',
                              gap: { xs: 1.5, sm: 0 }
                            }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                    <AutoAwesomeIcon sx={{ color: 'primary.main', fontSize: { xs: '1.2rem', sm: '1.5rem' } }} />
                                    <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>AI Analysis</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Box sx={{ textAlign: { xs: 'left', sm: 'right' } }}>
                                        <Typography variant="caption" sx={{ fontSize: { xs: '0.65rem', sm: '0.75rem' } }} display="block" color="text.secondary" gutterBottom>MODEL CONFIDENCE</Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <LinearProgress 
                                                variant="determinate" 
                                                value={lastResult.confidence * 100} 
                                                color={isLowConfidence ? 'warning' : 'primary'}
                                                sx={{ width: { xs: 60, sm: 80 }, height: 6, borderRadius: 2 }} 
                                            />
                                            <Typography variant="body2" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }} fontWeight={700}>{(lastResult.confidence * 100).toFixed(0)}%</Typography>
                                        </Box>
                                    </Box>
                                    <IconButton 
                                        size="small" 
                                        onClick={() => exportAnalysisPDF(lastResult)}
                                        sx={{ bgcolor: 'action.hover', '&:hover': { bgcolor: 'action.selected' } }}
                                    >
                                        <DownloadIcon sx={{ fontSize: { xs: 16, sm: 20 } }} />
                                    </IconButton>
                                </Box>
                            </Box>
                            
                            {/* Low Confidence Warning */}
                            {isLowConfidence && (
                              <Alert severity="warning" sx={{ mx: 2, mt: 1.5, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                                Low confidence result ({(lastResult.confidence * 100).toFixed(0)}%). Consider manual review.
                              </Alert>
                            )}
                            
                            {/* Badges Row */}
                             <Box sx={{ 
                               px: { xs: 1.5, sm: 2 }, 
                               py: 1.5, 
                               bgcolor: 'background.default', 
                               borderBottom: '1px solid',
                               borderColor: 'divider', 
                               display: 'flex', 
                               flexWrap: 'wrap',
                               gap: { xs: 1, sm: 2 }, 
                               width: '100%' 
                             }}>
                                <Chip 
                                  label={lastResult.urgency.toUpperCase()} 
                                  sx={{ 
                                    bgcolor: getUrgencyColor(lastResult.urgency), 
                                    color: '#fff', 
                                    fontWeight: 700,
                                    fontSize: { xs: '0.7rem', sm: '0.8125rem' }
                                  }} 
                                />
                                <Chip 
                                  label={lastResult.category} 
                                  variant="outlined" 
                                  sx={{ 
                                    bgcolor: 'background.paper', 
                                    fontWeight: 600,
                                    fontSize: { xs: '0.7rem', sm: '0.8125rem' }
                                  }} 
                                />
                             </Box>

                            {/* Content */}
                            <Box sx={{ 
                              p: { xs: 1.5, sm: 2 }, 
                              flexGrow: 1, 
                              overflowY: 'auto', 
                              wordBreak: 'break-word', 
                              width: '100%' 
                            }}>
                                
                                <Box sx={{ mb: { xs: 2, sm: 3 } }}>
                                    <Typography 
                                      variant="caption" 
                                      fontWeight={700} 
                                      color="text.secondary" 
                                      gutterBottom 
                                      display="block"
                                      sx={{ fontSize: { xs: '0.65rem', sm: '0.75rem' } }}
                                    >
                                      SUBJECT ANALYSIS
                                    </Typography>
                                    <Paper variant="outlined" sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: 'background.default', border: 'none' }}>
                                        <Typography 
                                          variant="body1" 
                                          fontWeight={500} 
                                          sx={{ 
                                            wordBreak: 'break-word',
                                            fontSize: { xs: '0.85rem', sm: '1rem' }
                                          }}
                                        >
                                            {renderHeatmap(lastResult.subject, lastResult.xai_highlights)}
                                        </Typography>
                                    </Paper>
                                </Box>

                                <Box>
                                    <Typography 
                                      variant="caption" 
                                      fontWeight={700} 
                                      color="text.secondary" 
                                      gutterBottom 
                                      display="block"
                                      sx={{ fontSize: { xs: '0.65rem', sm: '0.75rem' } }}
                                    >
                                      CONTENT ANALYSIS
                                    </Typography>
                                    <Paper variant="outlined" sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: 'background.default', border:'none' }}>
                                        <Typography 
                                          variant="body2" 
                                          sx={{ 
                                            lineHeight: 1.8, 
                                            color: 'text.secondary', 
                                            wordBreak: 'break-word', 
                                            whiteSpace:'wrap',
                                            fontSize: { xs: '0.8rem', sm: '0.875rem' }
                                          }}
                                        >
                                            {renderHeatmap(lastResult.text, lastResult.xai_highlights)}
                                        </Typography>
                                    </Paper>
                                </Box>
                            </Box>
                        </>
                    )}
                </Paper>
            </Grid>
        </Grid>
    </Container>
  );
}

export default Analysis;
