import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

/**
 * Export a single analysis result to PDF
 */
export const exportAnalysisPDF = (result) => {
    try {
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.getWidth();
        
        // Header
        doc.setFontSize(20);
        doc.setTextColor(15, 23, 42);
        doc.text('Email Analysis Report', pageWidth / 2, 20, { align: 'center' });
        
        // Timestamp
        doc.setFontSize(10);
        doc.setTextColor(100, 116, 139);
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, 28, { align: 'center' });
        
        // Divider
        doc.setDrawColor(226, 232, 240);
        doc.line(20, 35, pageWidth - 20, 35);
        
        // Classification Results
        doc.setFontSize(14);
        doc.setTextColor(15, 23, 42);
        doc.text('Classification Results', 20, 48);
        
        // Results table
        autoTable(doc, {
            startY: 55,
            head: [['Field', 'Value']],
            body: [
                ['Subject', result?.subject || 'No Subject'],
                ['Category', result?.category || 'Unknown'],
                ['Urgency', result?.urgency || 'Unknown'],
                ['Confidence', `${((result?.confidence || 0) * 100).toFixed(1)}%`],
                ['Analyzed At', result?.timestamp || new Date().toLocaleTimeString()]
            ],
            theme: 'striped',
            headStyles: { fillColor: [15, 23, 42] },
            styles: { fontSize: 11 },
            columnStyles: { 0: { fontStyle: 'bold', cellWidth: 40 } }
        });

        // Email Content Section
        const finalY = doc.lastAutoTable?.finalY || 100;
        const contentY = finalY + 15;
        doc.setFontSize(14);
        doc.setTextColor(15, 23, 42);
        doc.text('Email Content', 20, contentY);
        
        doc.setFontSize(10);
        doc.setTextColor(71, 85, 105);
        
        // Wrap long text
        const contentLines = doc.splitTextToSize(result?.text || 'No content', pageWidth - 40);
        doc.text(contentLines, 20, contentY + 10);
        
        // Footer
        doc.setFontSize(8);
        doc.setTextColor(148, 163, 184);
        doc.text(
            'Enterprise Email Classifier - AI Analysis Report',
            pageWidth / 2,
            doc.internal.pageSize.getHeight() - 10,
            { align: 'center' }
        );
        
        // Save
        doc.save(`email-analysis-${Date.now()}.pdf`);
    } catch (error) {
        console.error('PDF Export Error:', error);
        alert('Failed to export PDF. Please try again.');
    }
};

/**
 * Export history to PDF table
 */
export const exportHistoryPDF = (history, filters = {}) => {
    try {
        if (!history || history.length === 0) {
            alert('No history data to export.');
            return;
        }
        
        const doc = new jsPDF('landscape');
        const pageWidth = doc.internal.pageSize.getWidth();
        
        // Header
        doc.setFontSize(20);
        doc.setTextColor(15, 23, 42);
        doc.text('Email Analysis History', pageWidth / 2, 20, { align: 'center' });
        
        // Meta info
        doc.setFontSize(10);
        doc.setTextColor(100, 116, 139);
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, 28, { align: 'center' });
        doc.text(`Total Records: ${history.length}`, pageWidth / 2, 34, { align: 'center' });
        
        // Table
        const tableData = history.map(item => [
            item.timestamp || '-',
            (item.subject || 'No Subject').substring(0, 50),
            item.category || 'Unknown',
            item.urgency || 'Unknown',
            `${((item.confidence || 0) * 100).toFixed(0)}%`
        ]);
        
        autoTable(doc, {
            startY: 48,
            head: [['Time', 'Subject', 'Category', 'Urgency', 'Confidence']],
            body: tableData,
            theme: 'striped',
            headStyles: { 
                fillColor: [15, 23, 42],
                fontSize: 11
            },
            styles: { 
                fontSize: 9,
                cellPadding: 4
            }
        });
        
        // Footer
        doc.setFontSize(8);
        doc.setTextColor(148, 163, 184);
        doc.text(
            'Enterprise Email Classifier',
            pageWidth / 2,
            doc.internal.pageSize.getHeight() - 10,
            { align: 'center' }
        );
        
        // Save
        doc.save(`email-history-${Date.now()}.pdf`);
    } catch (error) {
        console.error('PDF Export Error:', error);
        alert('Failed to export PDF. Please try again.');
    }
};

/**
 * Export analytics summary to PDF
 */
export const exportAnalyticsPDF = (stats) => {
    try {
        if (!stats || stats.total === 0) {
            alert('No analytics data to export.');
            return;
        }
        
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.getWidth();
        
        // Header
        doc.setFontSize(20);
        doc.setTextColor(15, 23, 42);
        doc.text('Analytics Summary Report', pageWidth / 2, 20, { align: 'center' });
        
        doc.setFontSize(10);
        doc.setTextColor(100, 116, 139);
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, 28, { align: 'center' });
        
        // Summary stats
        doc.setFontSize(14);
        doc.setTextColor(15, 23, 42);
        doc.text('Overview', 20, 45);
        
        autoTable(doc, {
            startY: 52,
            head: [['Metric', 'Value']],
            body: [
                ['Total Emails Analyzed', (stats.total || 0).toString()],
                ['High Urgency', (stats.urgencyBreakdown?.High || 0).toString()],
                ['Medium Urgency', (stats.urgencyBreakdown?.Medium || 0).toString()],
                ['Low Urgency', (stats.urgencyBreakdown?.Low || 0).toString()],
                ['Average Confidence', `${((stats.avgConfidence || 0) * 100).toFixed(1)}%`]
            ],
            theme: 'striped',
            headStyles: { fillColor: [15, 23, 42] }
        });
        
        // Category breakdown
        if (stats.categoryBreakdown && Object.keys(stats.categoryBreakdown).length > 0) {
            const catY = (doc.lastAutoTable?.finalY || 100) + 15;
            doc.setFontSize(14);
            doc.text('Category Distribution', 20, catY);
            
            const catData = Object.entries(stats.categoryBreakdown).map(([cat, count]) => [cat, count.toString()]);
            
            autoTable(doc, {
                startY: catY + 7,
                head: [['Category', 'Count']],
                body: catData,
                theme: 'striped',
                headStyles: { fillColor: [15, 23, 42] }
            });
        }
        
        doc.save(`analytics-report-${Date.now()}.pdf`);
    } catch (error) {
        console.error('PDF Export Error:', error);
        alert('Failed to export PDF. Please try again.');
    }
};
