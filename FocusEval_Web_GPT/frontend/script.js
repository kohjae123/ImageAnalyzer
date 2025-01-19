document.getElementById('folderPath').addEventListener('input', getExtensions);

async function getExtensions() {
    const folderPath = document.getElementById('folderPath').value;
    if (!folderPath) return;

    try {
        const response = await fetch(`/get_extensions?folder_path=${encodeURIComponent(folderPath)}`);
        const data = await response.json();
        
        const extensionList = document.getElementById('extensionList');
        extensionList.innerHTML = '';
        
        if (data.extensions && data.extensions.length > 0) {
            data.extensions.forEach(ext => {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `ext-${ext}`;
                checkbox.value = ext;
                
                const label = document.createElement('label');
                label.htmlFor = `ext-${ext}`;
                label.textContent = ext;
                
                extensionList.appendChild(checkbox);
                extensionList.appendChild(label);
            });
            
            document.getElementById('algorithmSection').style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const folderPath = document.getElementById('folderPath').value;
    const algorithm = document.getElementById('algorithmSelect').value;
    const selectedExtensions = Array.from(document.querySelectorAll('#extensionList input[type="checkbox"]:checked'))
        .map(cb => cb.value);

    if (!folderPath || selectedExtensions.length === 0) {
        alert('폴더 경로와 확장자를 선택해주세요.');
        return;
    }

    try {
        // 버튼 상태 변경
        analyzeBtn.textContent = '분석 중...';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('folder_path', folderPath);
        formData.append('extensions', JSON.stringify(selectedExtensions));
        formData.append('algorithm', algorithm);

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert('오류 발생: ' + result.error);
        } else {
            alert(`분석 완료! ${result.results.length}개 파일을 처리했습니다.`);
            console.log('분석 결과:', result.results);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('분석 중 오류가 발생했습니다.');
    } finally {
        // 버튼 상태 복구
        analyzeBtn.textContent = '분석 시작';
        analyzeBtn.disabled = false;
    }
});
