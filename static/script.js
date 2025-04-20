
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const streamSourceEl = document.getElementById('stream-source');
    const serverPortEl = document.getElementById('server-port');
    const model1Toggle = document.getElementById('model1');
    const model2Toggle = document.getElementById('model2');
    const model3Toggle = document.getElementById('model3');
    const runScriptBtn = document.getElementById('run-script');
    const refreshStreamBtn = document.getElementById('refresh-stream');
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    
    // Load stream info
    fetchStreamInfo();
    
    // Set up event listeners
    model1Toggle.addEventListener('change', () => toggleModel(0, model1Toggle.checked));
    model2Toggle.addEventListener('change', () => toggleModel(1, model2Toggle.checked));
    model3Toggle.addEventListener('change', () => toggleModel(2, model3Toggle.checked));
    runScriptBtn.addEventListener('click', runMainPy);
    refreshStreamBtn.addEventListener('click', refreshStream);
    
    // Functions
    async function fetchStreamInfo() {
        try {
            const response = await fetch('/stream-info');
            const data = await response.json();
            
            streamSourceEl.textContent = data.stream_source;
            serverPortEl.textContent = data.server_port;
            
            // Update model toggles based on active state
            if (data.models_active) {
                model1Toggle.checked = data.models_active[0];
                model2Toggle.checked = data.models_active[1];
                model3Toggle.checked = data.models_active[2];
            }
        } catch (error) {
            showToast('Error fetching stream info: ' + error.message);
        }
    }
    
    async function toggleModel(modelId, active) {
        try {
            const response = await fetch(`/toggle-model/${modelId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ active })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showToast(`Model ${modelId + 1} ${active ? 'activated' : 'deactivated'}`);
            } else {
                showToast(`Error: ${data.error}`);
                // Reset the toggle to its previous state
                const toggleEl = document.getElementById(`model${modelId + 1}`);
                if (toggleEl) toggleEl.checked = !active;
            }
        } catch (error) {
            showToast('Error: ' + error.message);
        }
    }
    
    let chart;

    async function fetchAndRenderAnalytics() {
        const res = await fetch("/analytics");
        const data = await res.json();

        const labels = Object.keys(data);
        const values = Object.values(data);

        if (!chart) {
            const ctx = document.getElementById('detectionChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Detections',
                        data: values,
                        backgroundColor: 'rgba(52, 152, 219, 0.6)'
                    }]
                }
            });
        } else {
            chart.data.labels = labels;
            chart.data.datasets[0].data = values;
            chart.update();
        }
    }

    // Call this every few seconds
    setInterval(fetchAndRenderAnalytics, 3000);


    async function runMainPy() {
        try {
            runScriptBtn.disabled = true;
            
            const response = await fetch('/run-main-py', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                showToast('main.py started successfully');
            } else {
                showToast(`Error: ${data.error}`);
            }
        } catch (error) {
            showToast('Error: ' + error.message);
        } finally {
            runScriptBtn.disabled = false;
        }
    }
    
    async function refreshStream() {
        try {
            refreshStreamBtn.disabled = true;
            
            const response = await fetch('/refresh-stream', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                showToast('Stream refreshed');
            } else {
                showToast(`Error: ${data.error}`);
            }
        } catch (error) {
            showToast('Error: ' + error.message);
        } finally {
            refreshStreamBtn.disabled = false;
        }
    }
    
    function showToast(message) {
        toastMessage.textContent = message;
        toast.classList.remove('hidden');
        
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }
});
        