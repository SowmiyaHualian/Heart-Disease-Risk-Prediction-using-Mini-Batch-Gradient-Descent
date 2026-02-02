const API_URL = "http://127.0.0.1:8000/predict";

document.getElementById("predictForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const data = {
    age: Number(age.value),
    sex: Number(sex.value),
    cp: Number(cp.value),
    trestbps: Number(trestbps.value),
    chol: Number(chol.value),
    fbs: Number(fbs.value),
    restecg: Number(restecg.value),
    thalach: Number(thalach.value),
    exang: Number(exang.value),
    oldpeak: Number(oldpeak.value),
    slope: Number(slope.value),
    ca: Number(ca.value),
    thal: Number(thal.value)
  };

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await res.json();

  const output = document.getElementById("result");
  output.classList.remove("hidden", "low", "moderate", "high");

  let levelClass = result.risk_level.toLowerCase();
  output.classList.add(levelClass);

  output.innerHTML = `
    <strong>Risk Level:</strong> ${result.risk_level}<br>
    <strong>Risk Probability:</strong> ${result.risk_probability}
  `;
});
