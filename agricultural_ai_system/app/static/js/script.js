$(document).ready(function() {
    // Handle form submission
    $('#analysisForm').on('submit', function(e) {
        e.preventDefault();
        
        const analysisType = $('#analysisType').val();
        const imageFile = $('#imageUpload')[0].files[0];
        
        if (!analysisType || !imageFile) {
            alert('Please select both analysis type and upload an image');
            return;
        }
        
        // Show loading state
        const analyzeBtn = $('#analyzeBtn');
        const originalText = analyzeBtn.html();
        analyzeBtn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...');
        
        // Display original image
        const reader = new FileReader();
        reader.onload = function(e) {
            $('#originalImage').attr('src', e.target.result);
        };
        reader.readAsDataURL(imageFile);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('analysis_type', analysisType);
        
        // Send AJAX request
        $.ajax({
            url: '/analyze',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                displayResults(response, analysisType);
                $('#resultsSection').show();
                $('html, body').animate({
                    scrollTop: $('#resultsSection').offset().top - 20
                }, 500);
            },
            error: function(xhr, status, error) {
                console.error(error);
                alert('An error occurred during analysis. Please try again.');
            },
            complete: function() {
                analyzeBtn.prop('disabled', false).html(originalText);
            }
        });
    });
    
    // Display analysis results
    function displayResults(data, analysisType) {
        const resultsContainer = $('#analysisResults');
        resultsContainer.empty();
        
        if (data.error) {
            resultsContainer.append(`
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${data.error}
                </div>
            `);
            return;
        }
        
        switch(analysisType) {
            case 'crop':
                displayCropResults(data, resultsContainer);
                break;
            case 'disease':
                displayDiseaseResults(data, resultsContainer);
                break;
            case 'nutrient':
                displayNutrientResults(data, resultsContainer);
                break;
            case 'pest':
                displayPestResults(data, resultsContainer);
                break;
            case 'soil':
                displaySoilResults(data, resultsContainer);
                break;
            case 'weed':
                displayWeedResults(data, resultsContainer);
                break;
            default:
                resultsContainer.append(`
                    <div class="alert alert-warning">
                        Unknown analysis type
                    </div>
                `);
        }
    }
    
    function displayCropResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Crop Class:</div>
                <div class="result-value">${data.crop_class || 'Unknown'}</div>
            </div>
            <div class="result-item">
                <div class="result-title">Health Status:</div>
                <div class="result-value">
                    <span class="health-status ${data.health_status ? data.health_status.toLowerCase() : ''}">
                        ${data.health_status || 'Unknown'}
                    </span>
                </div>
            </div>
            <div class="result-item">
                <div class="result-title">Color Analysis:</div>
                <div class="result-value">
                    <div>RGB: ${data.color_analysis?.avg_rgb?.join(', ') || 'N/A'}</div>
                    <div>Green Ratio: ${data.color_analysis?.green_ratio || 'N/A'}</div>
                </div>
            </div>
        `);
        
        if (data.segmented_image) {
            container.append(`
                <div class="result-item">
                    <div class="result-title">Segmentation Visualization:</div>
                    <img src="${data.segmented_image}" class="visualization-img" alt="Segmentation">
                </div>
            `);
        }
    }
    
    function displayDiseaseResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Disease Detected:</div>
                <div class="result-value">
                    ${data.disease_detected ? 'Yes' : 'No'}
                    ${data.disease_type && data.disease_type.toLowerCase() !== 'healthy' ? `(${data.disease_type})` : ''}
                </div>
            </div>
            <div class="result-item">
                <div class="result-title">Confidence:</div>
                <div class="result-value">${data.confidence ? (data.confidence * 100).toFixed(2) + '%' : 'N/A'}</div>
            </div>
        `);
        
        if (data.recommendations && data.recommendations.length > 0) {
            const recommendationsHtml = data.recommendations.map(rec => `<li>${rec}</li>`).join('');
            container.append(`
                <div class="result-item">
                    <div class="result-title">Recommendations:</div>
                    <ul class="result-value">${recommendationsHtml}</ul>
                </div>
            `);
        }
        
        if (data.visualization) {
            container.append(`
                <div class="result-item">
                    <div class="result-title">Visualization:</div>
                    <img src="data:image/png;base64,${data.visualization}" class="visualization-img" alt="Disease Visualization">
                </div>
            `);
        }
    }
    
    function displayNutrientResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Primary Deficiency:</div>
                <div class="result-value">${data.primary_deficiency || 'Unknown'}</div>
            </div>
            <div class="result-item">
                <div class="result-title">Confidence:</div>
                <div class="result-value">${data.confidence ? data.confidence.toFixed(2) + '%' : 'N/A'}</div>
            </div>
        `);
        
        if (data.recommendations && data.recommendations.length > 0) {
            const recommendationsHtml = data.recommendations.map(rec => `<li>${rec}</li>`).join('');
            container.append(`
                <div class="result-item">
                    <div class="result-title">Recommendations:</div>
                    <ul class="result-value">${recommendationsHtml}</ul>
                </div>
            `);
        }
        
        if (data.visualization) {
            container.append(`
                <div class="result-item">
                    <div class="result-title">Analysis Visualization:</div>
                    <img src="data:image/png;base64,${data.visualization}" class="visualization-img" alt="Nutrient Visualization">
                </div>
            `);
        }
    }
    
    function displayPestResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Pest Detected:</div>
                <div class="result-value">
                    ${data.pest_detected ? 'Yes' : 'No'}
                    ${data.pest_type && data.pest_type.toLowerCase() !== 'healthy' ? `(${data.pest_type})` : ''}
                </div>
            </div>
            <div class="result-item">
                <div class="result-title">Confidence:</div>
                <div class="result-value">${data.confidence ? (data.confidence * 100).toFixed(2) + '%' : 'N/A'}</div>
            </div>
        `);
        
        if (data.all_predictions && Object.keys(data.all_predictions).length > 0) {
            let predictionsHtml = '';
            for (const [pest, confidence] of Object.entries(data.all_predictions)) {
                predictionsHtml += `<li>${pest}: ${(confidence * 100).toFixed(2)}%</li>`;
            }
            container.append(`
                <div class="result-item">
                    <div class="result-title">All Predictions:</div>
                    <ul class="result-value">${predictionsHtml}</ul>
                </div>
            `);
        }
    }
    
    function displaySoilResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Classification Model Prediction:</div>
                <div class="result-value">${data.classification_prediction || 'Unknown'}</div>
            </div>
            <div class="result-item">
                <div class="result-title">Deep Learning Prediction:</div>
                <div class="result-value">${data.deep_learning_prediction || 'Unknown'}</div>
            </div>
        `);
    }
    
    function displayWeedResults(data, container) {
        container.append(`
            <div class="result-item">
                <div class="result-title">Weed Detected:</div>
                <div class="result-value">
                    ${data.weed_present ? 'Yes' : 'No'}
                </div>
            </div>
            <div class="result-item">
                <div class="result-title">Confidence:</div>
                <div class="result-value">${data.confidence ? (data.confidence * 100).toFixed(2) + '%' : 'N/A'}</div>
            </div>
        `);
        
        if (data.visualization) {
            container.append(`
                <div class="result-item">
                    <div class="result-title">Weed Visualization:</div>
                    <img src="${data.visualization}" class="visualization-img" alt="Weed Visualization">
                </div>
            `);
        }
    }
});