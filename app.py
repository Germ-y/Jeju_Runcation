from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

coeffs = {
    'const': 2.625250871672329,
    '감귤_과일소비량': 18.548356714914902,
    '노지_온주밀감_생산성': 6.340148975053342,
    'WS_AVG': 9.506145364385503,
    '감귤_과일수출량': -0.0009609116874985356,
    '하우스_온주밀감_생산성': 0.13114849779832138,
    '과실주스_과일수출량': 0.0007250232358434108,
    'RN_DAY': -0.15796225429840638,
    '월동_온주밀감_생산성': 0.4571769085408688,
    '복숭아_과일소비량': -42.16527122509427,
    '감귤_면적': 13.296038440072467
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    감귤_과일소비량: float
    노지_온주밀감_생산성: float
    WS_AVG: float
    감귤_과일수출량: float
    하우스_온주밀감_생산성: float
    과실주스_과일수출량: float
    RN_DAY: float
    월동_온주밀감_생산성: float
    복숭아_과일소비량: float
    감귤_면적: float

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/predict")
def predict(data: InputData):
    y_pred = coeffs['const']
    y_pred += coeffs['감귤_과일소비량'] * data.감귤_과일소비량
    y_pred += coeffs['노지_온주밀감_생산성'] * data.노지_온주밀감_생산성
    y_pred += coeffs['WS_AVG'] * data.WS_AVG
    y_pred += coeffs['감귤_과일수출량'] * data.감귤_과일수출량
    y_pred += coeffs['하우스_온주밀감_생산성'] * data.하우스_온주밀감_생산성
    y_pred += coeffs['과실주스_과일수출량'] * data.과실주스_과일수출량
    y_pred += coeffs['RN_DAY'] * data.RN_DAY
    y_pred += coeffs['월동_온주밀감_생산성'] * data.월동_온주밀감_생산성
    y_pred += coeffs['복숭아_과일소비량'] * data.복숭아_과일소비량
    y_pred += coeffs['감귤_면적'] * data.감귤_면적
    return {"y_pred": y_pred}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
