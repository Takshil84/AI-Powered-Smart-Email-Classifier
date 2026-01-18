import React, { useState, useMemo } from 'react';
import { 
  Box, Container, Paper, Typography, Button, 
  Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  IconButton, Collapse, Grid, TextField, MenuItem, InputAdornment,
  Tooltip, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions,
  Pagination, Stack
} from '@mui/material';
import { 
    DeleteOutline as DeleteIcon,
    History as HistoryIcon,
    KeyboardArrowDown as KeyboardArrowDownIcon,
    KeyboardArrowUp as KeyboardArrowUpIcon,
    Search as SearchIcon,
    FilterList as FilterIcon,
    ContentCopy as CopyIcon,
    AutoAwesome as AutoAwesomeIcon,
    Warning as WarningIcon,
    Download as DownloadIcon
} from '@mui/icons-material';
import { renderHeatmap } from '../utils/heatmap';
import { exportHistoryPDF } from '../utils/pdfExport';

// -- Row Component --
function HistoryRow({ row, getUrgencyColor, onDelete }) {
    const [open, setOpen] = useState(false);

    const handleCopy = (e) => {
        e.stopPropagation();
        const text = `Subject: ${row.subject}\nUrgency: ${row.urgency}\nCategory: ${row.category}\nConfidence: ${(row.confidence*100).toFixed(0)}%\n\nContent Analysis:\n${row.text}`;
        navigator.clipboard.writeText(text);
        // Could show a toast here if we had a snackbar context
    };

    const handleDelete = (e) => {
        e.stopPropagation();
        onDelete(row.id);
    };

    return (
        <>
            <TableRow hover sx={{ '& > *': { borderBottom: 'unset' }, cursor: 'pointer' }} onClick={() => setOpen(!open)}>
                <TableCell width={40} sx={{ p: { xs: 0.5, sm: 1 } }}>
                    <IconButton aria-label="expand row" size="small" onClick={(e) => {
                        e.stopPropagation();
                        setOpen(!open);
                    }}>
                        {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                    </IconButton>
                </TableCell>
                <TableCell sx={{ color: 'text.secondary', fontSize: { xs: '0.75rem', sm: '0.85rem' }, display: { xs: 'none', md: 'table-cell' } }}>{row.timestamp}</TableCell>
                <TableCell>
                    <Typography variant="body2" fontWeight="600" sx={{ fontSize: { xs: '0.8rem', sm: '0.9rem' } }}>{row.subject || "No Subject"}</Typography>
                </TableCell>
                <TableCell>
                    <Chip 
                        label={row.urgency} size="small" 
                        sx={{ 
                            bgcolor: getUrgencyColor(row.urgency), color: '#fff', fontWeight: 700,
                            height: { xs: 20, sm: 24 }, fontSize: { xs: '0.65rem', sm: '0.75rem' }
                        }} 
                    />
                </TableCell>
                <TableCell>
                    <Chip label={row.category} size="small" variant="outlined" sx={{ fontWeight: 600, height: { xs: 20, sm: 24 }, fontSize: { xs: '0.65rem', sm: '0.75rem' } }} />
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: 600, color: 'text.secondary', display: { xs: 'none', md: 'table-cell' }, fontSize: { xs: '0.75rem', sm: '0.85rem' } }}>{(row.confidence*100).toFixed(0)}%</TableCell>
                <TableCell align="right" sx={{ p: { xs: 0.5, sm: 1 } }} onClick={(e) => e.stopPropagation()}>
                    <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                        <Tooltip title="Copy Details">
                            <IconButton size="small" onClick={handleCopy}><CopyIcon sx={{ fontSize: { xs: 16, sm: 20 } }} /></IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Item">
                            <IconButton size="small" color="error" onClick={handleDelete}><DeleteIcon sx={{ fontSize: { xs: 16, sm: 20 } }} /></IconButton>
                        </Tooltip>
                    </Stack>
                </TableCell>
            </TableRow>
            <TableRow>
                <TableCell style={{ paddingBottom: 0, paddingTop: 0, backgroundColor: 'var(--mui-palette-action-hover)' }} colSpan={7}>
                    <Collapse in={open} timeout="auto" unmountOnExit>
                        <Paper 
                            sx={{ 
                                m: 2, p: 3, 
                                border: '1px solid',
                                borderColor: 'divider', 
                                borderRadius: 2, 
                                borderLeft: `4px solid ${getUrgencyColor(row.urgency)}`
                            }} 
                            elevation={0}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                                <AutoAwesomeIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                                <Typography variant="subtitle2" fontWeight={700}>AI Interpretation</Typography>
                            </Box>
                            
                            <Grid container spacing={3}>
                                <Grid item xs={12}>
                                    <Box sx={{ mb: 2 }}>
                                        <Typography variant="caption" fontWeight={700} color="text.secondary" display="block" gutterBottom>SUBJECT ANALYSIS</Typography>
                                        <Typography variant="body2" fontWeight={500}>
                                            {renderHeatmap(row.subject, row.xai_highlights)}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography variant="caption" fontWeight={700} color="text.secondary" display="block" gutterBottom>CONTENT ANALYSIS</Typography>
                                        <Typography variant="body2" sx={{ lineHeight: 1.7, color: 'text.secondary' }}>
                                            {renderHeatmap(row.text, row.xai_highlights)}
                                        </Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                        </Paper>
                    </Collapse>
                </TableCell>
            </TableRow>
        </>
    );
}

// -- Main History Page --
function History({ history, clearHistory, deleteHistoryItem, getUrgencyColor }) {
    const [searchTerm, setSearchTerm] = useState("");
    const [filterUrgency, setFilterUrgency] = useState("All");
    const [filterCategory, setFilterCategory] = useState("All");
    const [confirmOpen, setConfirmOpen] = useState(false);

    // Derived State
    const filteredHistory = useMemo(() => {
        return history.filter(item => {
            const matchesSearch = 
                (item.subject || "").toLowerCase().includes(searchTerm.toLowerCase()) || 
                (item.text || "").toLowerCase().includes(searchTerm.toLowerCase());
            
            const matchesUrgency = filterUrgency === "All" || item.urgency === filterUrgency;
            const matchesCategory = filterCategory === "All" || item.category === filterCategory;

            return matchesSearch && matchesUrgency && matchesCategory;
        });
    }, [history, searchTerm, filterUrgency, filterCategory]);

    const categories = ["All", ...new Set(history.map(h => h.category))];

    return (
        <Container 
            maxWidth="xl" 
            sx={{ 
                mt: { xs: 2, sm: 4 }, 
                height: { xs: 'auto', md: 'calc(100vh - 120px)' }, 
                minHeight: { xs: 'calc(100vh - 150px)', md: 'auto' },
                display: 'flex', 
                flexDirection: 'column', 
                pb: 2,
                px: { xs: 1, sm: 2, md: 3 }
            }}
        >
            
            {/* Header & Controls */}
            <Box sx={{ mb: { xs: 2, sm: 3 } }}>
                <Box sx={{ 
                    display: 'flex', 
                    flexDirection: { xs: 'column', sm: 'row' },
                    justifyContent: 'space-between', 
                    alignItems: { xs: 'flex-start', sm: 'center' }, 
                    mb: { xs: 2, sm: 3 },
                    gap: { xs: 2, sm: 0 }
                }}>
                    <Box>
                        <Typography variant="h5" sx={{ fontSize: { xs: '1.25rem', sm: '1.5rem' } }} color="text.primary">Session History</Typography>
                        <Typography variant="body2" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }} color="text.secondary">Review, filter, and manage your analyzed emails.</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button 
                            startIcon={<DownloadIcon />}
                            variant="outlined"
                            onClick={() => exportHistoryPDF(filteredHistory, { urgency: filterUrgency, category: filterCategory, search: searchTerm })}
                            disabled={filteredHistory.length === 0}
                            size="small"
                            sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
                        >
                            Export PDF
                        </Button>
                        <Button 
                            startIcon={<DeleteIcon />} 
                            color="error" 
                            variant="outlined"
                            onClick={() => setConfirmOpen(true)}
                            disabled={history.length === 0}
                            size="small"
                            sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
                        >
                            Clear All
                        </Button>
                    </Box>
                </Box>

                {/* Filters Row */}
                <Paper sx={{ 
                    p: { xs: 1.5, sm: 2 }, 
                    mb: { xs: 2, sm: 3 }, 
                    display: 'flex', 
                    gap: { xs: 1.5, sm: 2 }, 
                    alignItems: 'center', 
                    flexWrap: 'wrap' 
                }}>
                    <TextField 
                        size="small" placeholder="Search logs..." 
                        value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><SearchIcon color="action" fontSize="small" /></InputAdornment>,
                            sx: { fontSize: { xs: '0.8rem', sm: '0.875rem' } }
                        }}
                        sx={{ 
                            minWidth: { xs: '100%', sm: 200, md: 250 }, 
                            flex: { xs: '1 1 100%', sm: '0 0 auto' },
                            '& .MuiOutlinedInput-root': { bgcolor: 'background.default' } 
                        }}
                    />
                    
                    <TextField 
                        select size="small" label="Urgency"
                        value={filterUrgency} onChange={e => setFilterUrgency(e.target.value)}
                        sx={{ 
                            minWidth: { xs: 'calc(50% - 6px)', sm: 130, md: 150 },
                            flex: { xs: '1 1 calc(50% - 6px)', sm: '0 0 auto' }
                        }}
                        InputProps={{ 
                            startAdornment: <FilterIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary', display: { xs: 'none', sm: 'block' } }} />,
                            sx: { fontSize: { xs: '0.8rem', sm: '0.875rem' } }
                        }}
                    >
                        <MenuItem value="All">All</MenuItem>
                        <MenuItem value="High">High</MenuItem>
                        <MenuItem value="Medium">Medium</MenuItem>
                        <MenuItem value="Low">Low</MenuItem>
                    </TextField>

                     <TextField 
                        select size="small" label="Category"
                        value={filterCategory} onChange={e => setFilterCategory(e.target.value)}
                        sx={{ 
                            minWidth: { xs: 'calc(50% - 6px)', sm: 130, md: 150 },
                            flex: { xs: '1 1 calc(50% - 6px)', sm: '0 0 auto' }
                        }}
                        InputProps={{ sx: { fontSize: { xs: '0.8rem', sm: '0.875rem' } } }}
                    >
                        {categories.map(cat => (
                            <MenuItem key={cat} value={cat}>{cat}</MenuItem>
                        ))}
                    </TextField>
                </Paper>
            </Box>

            {/* Content Table */}
            {filteredHistory.length > 0 ? (
                <TableContainer 
                    component={Paper} 
                    elevation={0} 
                    sx={{ 
                        flexGrow: 1, 
                        border: '1px solid',
                        borderColor: 'divider', 
                        borderRadius: { xs: 2, sm: 3 }, 
                        overflowX: 'auto'
                    }}
                >
                    <Table stickyHeader size="small" sx={{ minWidth: { xs: 500, sm: 700 } }}>
                        <TableHead>
                            <TableRow>
                                <TableCell width={40} sx={{ p: { xs: 0.5, sm: 1 } }} />
                                <TableCell sx={{ fontWeight: 600, display: { xs: 'none', md: 'table-cell' }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Time</TableCell>
                                <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Subject</TableCell>
                                <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Urgency</TableCell>
                                <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Category</TableCell>
                                <TableCell align="right" sx={{ fontWeight: 600, display: { xs: 'none', md: 'table-cell' }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Confidence</TableCell>
                                <TableCell align="right" sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.5, sm: 1 } }}>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredHistory.map((row) => (
                                <HistoryRow 
                                    key={row.id} 
                                    row={row} 
                                    getUrgencyColor={getUrgencyColor} 
                                    onDelete={deleteHistoryItem}
                                />
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            ) : (
                <Box sx={{ 
                    p: { xs: 4, sm: 8 }, 
                    textAlign: 'center', 
                    color: 'text.secondary', 
                    flexGrow: 1, 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    bgcolor: 'background.paper', 
                    borderRadius: { xs: 2, sm: 4 }, 
                    border: '1px dashed',
                    borderColor: 'divider'
                }}>
                    <HistoryIcon sx={{ fontSize: { xs: 36, sm: 48 }, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }} color="text.primary">No records found</Typography>
                    <Typography variant="body2" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>Try adjusting your filters or analyze new emails.</Typography>
                </Box>
            )}

            {/* Clear Confirmation Modal */}
            <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'error.main' }}>
                    <WarningIcon color="error" /> Confirm Action
                </DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Are you sure you want to clear your entire session history? This action cannot be undone.
                    </DialogContentText>
                </DialogContent>
                <DialogActions sx={{ p: 2, pt: 0 }}>
                    <Button onClick={() => setConfirmOpen(false)} color="inherit">Cancel</Button>
                    <Button onClick={() => { clearHistory(); setConfirmOpen(false); }} color="error" variant="contained" disableElevation>
                        Clear History
                    </Button>
                </DialogActions>
            </Dialog>

        </Container>
    );
}

export default History;
