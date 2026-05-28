const API = "https://nicho106.pythonanywhere.com";

const advice = {
low: [
    'Pertahankan pola makan sehat dengan gizi seimbang',
    'Olahraga rutin minimal 150 menit per minggu',
    'Cek gula darah secara berkala setahun sekali',
    'Hindari konsumsi gula berlebihan'
],
medium: [
    'Konsultasikan hasil ini dengan dokter atau tenaga medis',
    'Kurangi asupan karbohidrat sederhana dan gula tambahan',
    'Mulai olahraga terjadwal seperti jalan kaki, renang, atau bersepeda',
    'Pantau berat badan, tekanan darah, dan gula darah secara rutin'
],
high: [
    'Hindari minuman manis, makanan berlemak tinggi, dan alkohol',
    'Ikuti program manajemen berat badan jika BMI di atas normal',
    'Jangan tunda — penanganan dini sangat berpengaruh pada prognosis'
]
};

const feature = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
];

async function predict() {
    const btn = document.getElementById('predictBtn');
    const errorMsg  = document.getElementById('error-msg');
    const resultEl  = document.getElementById('result');
    errorMsg.style.display = 'none';
    resultEl.style.display = 'none';
    feature.forEach(f => document.getElementById(f).classList.remove('error'));
    const payload = {};
    let hasError = false;

    feature.forEach(f => {
      const el  = document.getElementById(f);
      const val = el.value.trim();
      if (val === '') {
        el.classList.add('error');
        hasError = true;
      } else {
        payload[f] = parseFloat(val);
      }
    });

    if (hasError) {
      showError('Lengkapi semua field yang ditandai merah terlebih dahulu.');
      return;
    }


    btn.textContent = 'Menganalisis...';
    btn.disabled = true;

    try {
      const res = await fetch(`${API}/predict`, {
        method:'POST',
        headers:{ 'Content-Type': 'application/json' },
        body:JSON.stringify(payload)
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || 'Server error');

      showResult(data);
    } 
    catch (err) {
      if (err.message.toLowerCase().includes('fetch')) {
        showError('Tidak dapat terhubung ke server. Pastikan app.py sudah berjalan di localhost:5000.');
      } else {
        showError(err.message);
      }
    } finally {
      btn.textContent = 'Analisis Sekarang';
      btn.disabled = false;
    }
}

function showResult(data) {
    const resultEl = document.getElementById('result');
    const level = data.risk_level;

    const bulet = { low: '🟢', medium: '🟡', high: '🔴' };
    const desc  = {
      low:'Pola hidup saat ini sudah cukup baik. Pertahankan!',
      medium:'Perlu perhatian lebih. Segera konsultasi ke dokter.',
      high:'Segera lakukan pemeriksaan medis lebih lanjut.'
    };

    document.getElementById('bulet').textContent = bulet[level];
    document.getElementById('resultTitle').textContent = data.risk_label;
    document.getElementById('resultdesc').textContent = desc[level];
    document.getElementById('probPct').textContent = `${data.probability}%`;

    document.getElementById('adviceList').innerHTML = advice[level].map(a => `<li>${a}</li>`).join('');

    resultEl.className = `result-section ${level}`;
    resultEl.style.display = 'block';

    
    const fill = document.getElementById('probFill');
    fill.style.width = '0%';
    setTimeout(() => { fill.style.width = `${data.probability}%`; }, 50);
}

function showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.style.display = 'block';
}


document.addEventListener('keydown', e => {
    if (e.key === 'Enter') predict();
});
