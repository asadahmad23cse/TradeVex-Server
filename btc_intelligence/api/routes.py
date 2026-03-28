from __future__ import annotations

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect

from btc_intelligence.service import AppRuntime


router = APIRouter()


def runtime(request: Request) -> AppRuntime:
    return request.app.state.runtime  # type: ignore[attr-defined]


@router.get('/signal')
async def get_signal(request: Request):
    return await runtime(request).latest_signal()


@router.get('/signal/history')
async def get_signal_history(request: Request):
    return await runtime(request).signal_history()


@router.get('/regime')
async def get_regime(request: Request):
    return await runtime(request).latest_regime()


@router.get('/features')
async def get_features(request: Request):
    return await runtime(request).current_features()


@router.get('/health')
async def get_health(request: Request):
    return await runtime(request).health()


@router.get('/models/performance')
async def get_models_performance(request: Request):
    return await runtime(request).model_performance()


@router.get('/monitoring/stats')
async def get_monitoring_stats(request: Request):
    return await runtime(request).monitoring_stats()


@router.get('/monitoring/edges')
async def get_monitoring_edges(request: Request):
    return await runtime(request).monitoring_edges()


@router.post('/models/retrain')
async def trigger_model_retrain(request: Request):
    payload = await request.json()
    return await runtime(request).trigger_model_retrain(payload)


@router.post('/paper_trade/open')
async def open_paper_trade(request: Request):
    payload = await request.json()
    return await runtime(request).open_paper_trade(payload)


@router.post('/paper_trade/close')
async def close_paper_trade(request: Request):
    payload = await request.json()
    return await runtime(request).close_paper_trade(payload)


@router.get('/market/klines')
async def get_market_klines(
    request: Request,
    timeframe: str = Query('15m', pattern='^(1m|5m|15m|1h|4h)$'),
    limit: int = Query(500, ge=10, le=5000),
):
    return await runtime(request).market_klines(timeframe=timeframe, limit=limit)


@router.get('/market/history')
async def get_market_history(
    request: Request,
    timeframe: str = Query('1d', pattern='^(1m|5m|15m|1h|4h|1d)$'),
    limit: int = Query(2000, ge=100, le=5000),
):
    return await runtime(request).market_history(timeframe=timeframe, limit=limit)


@router.websocket('/ws/live')
async def ws_live(websocket: WebSocket):
    rt: AppRuntime = websocket.app.state.runtime  # type: ignore[attr-defined]
    initial = await rt.latest_signal()
    await rt.hub.connect(websocket, initial_payload=initial)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await rt.hub.disconnect(websocket)
