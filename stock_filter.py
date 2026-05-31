# -*- coding: utf-8 -*-
"""
股票筛选工具
==============

筛选条件：
- 主板（沪市 60xxxx、深市 00xxxx）
- 当天涨幅 3-5%
- 市值 50-200 亿
- 板块资金净流入（如果数据源支持）
- 量比 2-4
- 换手率 5-10%

如果当天未开盘，自动使用最近交易日数据
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def is_main_board(code: str) -> bool:
    """
    判断是否为主板股票

    主板：
    - 沪市：60xxxx（科创板 688xxxx 除外）
    - 深市：00xxxx（创业板 300xxxx 除外）

    Args:
        code: 股票代码

    Returns:
        True 表示主板，False 表示非主板
    """
    code = str(code).strip()
    if len(code) >= 4 and (code.startswith("60") or code.startswith("00")):
        return True
    return False


def is_trading_day() -> bool:
    """
    判断今天是否为交易日

    简化判断：周一到周五为默认交易日
    实际应以交易所日历为准

    Returns:
        True 表示交易日，False 表示非交易日
    """
    today = datetime.now().weekday()
    return today < 5


def get_latest_trading_day() -> str:
    """
    获取最近交易日日期字符串

    Returns:
        格式为 YYYYMMDD 的日期字符串
    """
    today = datetime.now()
    if today.weekday() == 5:
        days_ago = 1
    elif today.weekday() == 6:
        days_ago = 2
    else:
        days_ago = 0
    trading_day = today - timedelta(days=days_ago)
    return trading_day.strftime("%Y%m%d")


def filter_stocks(
    min_change_pct: float = 3.0,
    max_change_pct: float = 5.0,
    min_market_cap: float = 50.0,
    max_market_cap: float = 200.0,
    min_volume_ratio: float = 2.0,
    max_volume_ratio: float = 4.0,
    min_turnover: float = 5.0,
    max_turnover: float = 10.0,
    sector_net_inflow_only: bool = True,
) -> Optional[pd.DataFrame]:
    """
    根据条件筛选股票

    Args:
        min_change_pct: 最小涨幅百分比
        max_change_pct: 最大涨幅百分比
        min_market_cap: 最小市值（亿元）
        max_market_cap: 最大市值（亿元）
        min_volume_ratio: 最小量比
        max_volume_ratio: 最大量比
        min_turnover: 最小换手率
        max_turnover: 最大换手率
        sector_net_inflow_only: 是否仅保留板块资金净流入的股票

    Returns:
        符合条件的股票 DataFrame，如果获取失败返回 None
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装，请运行: pip install akshare")
        return None

    trading_day = get_latest_trading_day()
    logger.info(f"使用交易日数据: {trading_day}")

    try:
        logger.info("正在获取 A 股实时行情数据...")
        import time
        last_error = None
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    logger.info(f"获取到 {len(df)} 只股票")
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"获取行情数据失败 (尝试 {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(3)
        else:
            logger.info("东方财富接口失败，尝试新浪财经接口...")
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_spot()
                    if df is not None and not df.empty:
                        logger.info(f"新浪接口获取到 {len(df)} 只股票")
                        break
                except Exception as e2:
                    last_error = e2
                    logger.warning(f"新浪接口失败 (尝试 {attempt + 1}/3): {e2}")
                    if attempt < 2:
                        time.sleep(3)
            else:
                logger.error(f"获取行情数据最终失败: {last_error}")
                return None
    except Exception as e:
        logger.error(f"获取行情数据失败: {e}")
        return None

    if df is None or df.empty:
        logger.error("行情数据为空")
        return None

    logger.info(f"数据列名: {list(df.columns)}")

    code_col = _find_column(df, ["代码", "code", "股票代码"])
    name_col = _find_column(df, ["名称", "name", "股票名称"])
    price_col = _find_column(df, ["最新价", "price", "lastPrice"])
    change_pct_col = _find_column(df, ["涨跌幅", "change_pct", "pct_change"])
    volume_ratio_col = _find_column(df, ["量比", "volume_ratio"])
    turnover_col = _find_column(df, ["换手率", "turnover", "turnover_rate"])
    market_cap_col = _find_column(df, ["总市值", "market_cap", "total_market_cap"])
    industry_col = _find_column(df, ["所属行业", "industry", "sector"])
    amount_col = _find_column(df, ["成交额", "amount"])

    if change_pct_col is None:
        logger.error("无法找到涨跌幅列")
        return None

    if market_cap_col is None:
        logger.warning("数据中无市值列，跳过市值筛选")

    if market_cap_col is None:
        use_market_cap_filter = False
    else:
        use_market_cap_filter = True

    df_filtered = df.copy()

    if code_col:
        df_filtered = df_filtered[df_filtered[code_col].apply(is_main_board)]
        logger.info(f"主板筛选后: {len(df_filtered)} 只")

    if change_pct_col:
        df_filtered[change_pct_col] = pd.to_numeric(df_filtered[change_pct_col], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered[change_pct_col] >= min_change_pct)
            & (df_filtered[change_pct_col] <= max_change_pct)
        ]
        logger.info(f"涨幅 {min_change_pct}-{max_change_pct}% 筛选后: {len(df_filtered)} 只")

    if market_cap_col:
        df_filtered[market_cap_col] = pd.to_numeric(df_filtered[market_cap_col], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered[market_cap_col] >= min_market_cap * 1e8)
            & (df_filtered[market_cap_col] <= max_market_cap * 1e8)
        ]
        logger.info(f"市值 {min_market_cap}-{max_market_cap}亿 筛选后: {len(df_filtered)} 只")

    if volume_ratio_col:
        df_filtered[volume_ratio_col] = pd.to_numeric(df_filtered[volume_ratio_col], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered[volume_ratio_col] >= min_volume_ratio)
            & (df_filtered[volume_ratio_col] <= max_volume_ratio)
        ]
        logger.info(f"量比 {min_volume_ratio}-{max_volume_ratio} 筛选后: {len(df_filtered)} 只")

    if turnover_col:
        df_filtered[turnover_col] = pd.to_numeric(df_filtered[turnover_col], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered[turnover_col] >= min_turnover)
            & (df_filtered[turnover_col] <= max_turnover)
        ]
        logger.info(f"换手率 {min_turnover}-{max_turnover}% 筛选后: {len(df_filtered)} 只")

    if sector_net_inflow_only and industry_col:
        try:
            logger.info("正在获取板块资金流向数据...")
            sector_flow = _get_sector_fund_flow()
            if sector_flow:
                net_inflow_sectors = set(sector_flow.keys())
                df_filtered = df_filtered[
                    df_filtered[industry_col].isin(net_inflow_sectors)
                ]
                logger.info(f"板块资金净流入筛选后: {len(df_filtered)} 只")
            else:
                logger.warning("板块资金流向数据获取失败，跳过该筛选条件")
        except Exception as e:
            logger.warning(f"板块资金流向筛选失败: {e}，跳过该筛选条件")

    if len(df_filtered) == 0:
        logger.warning("没有符合筛选条件的股票")
        return df_filtered

    display_cols = [col for col in [code_col, name_col, change_pct_col, volume_ratio_col, turnover_col, market_cap_col] if col]
    display_cols = list(dict.fromkeys(display_cols))

    result = df_filtered[display_cols].copy()

    if market_cap_col:
        result["市值（亿）"] = (result[market_cap_col] / 1e8).round(2)

    if change_pct_col:
        result["涨跌幅（%）"] = result[change_pct_col].round(2)

    if volume_ratio_col:
        result["量比"] = result[volume_ratio_col].round(2)

    if turnover_col:
        result["换手率（%）"] = result[turnover_col].round(2)

    cols_to_show = ["代码", "名称", "涨跌幅（%）", "量比", "换手率（%）", "市值（亿）"]
    result = result.rename(columns={
        code_col: "代码",
        name_col: "名称",
    })
    existing_cols = [c for c in cols_to_show if c in result.columns]
    result = result[existing_cols]

    return result


def _find_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """在 DataFrame 中查找存在的列名"""
    for name in names:
        if name in df.columns:
            return name
    return None


def _get_sector_fund_flow() -> Optional[Dict[str, float]]:
    """
    获取板块资金净流入数据

    Returns:
        板块名称到净流入金额的字典，单位为元
    """
    try:
        import akshare as ak

        df = ak.stock_sector_fund_flow_rank(indicator="今日")
        if df is None or df.empty:
            return None

        fund_flow_col = _find_column(df, ["今日主力净流入-净额", "主力净流入", "fund_flow"])
        sector_col = _find_column(df, ["板块名称", "名称", "sector", "name"])

        if fund_flow_col is None or sector_col is None:
            logger.warning(f"板块资金流向数据列不完整，可用列: {list(df.columns)}")
            return None

        result = {}
        for _, row in df.iterrows():
            sector = str(row[sector_col]).strip()
            flow = pd.to_numeric(row[fund_flow_col], errors="coerce")
            if flow > 0:
                result[sector] = flow

        return result

    except Exception as e:
        logger.warning(f"获取板块资金流向失败: {e}")
        return None


def print_results(df: pd.DataFrame) -> None:
    """打印筛选结果"""
    if df is None or df.empty:
        print("\n没有符合筛选条件的股票\n")
        return

    print(f"\n{'='*80}")
    print(f"筛选结果：共 {len(df)} 只股票符合条件")
    print(f"{'='*80}\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.colheader_justify", "center")
    print(df.to_string(index=False))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="根据条件筛选股票")
    parser.add_argument("--min-change", type=float, default=3.0, help="最小涨幅（%%），默认 3.0")
    parser.add_argument("--max-change", type=float, default=5.0, help="最大涨幅（%%），默认 5.0")
    parser.add_argument("--min-cap", type=float, default=50.0, help="最小市值（亿），默认 50.0")
    parser.add_argument("--max-cap", type=float, default=200.0, help="最大市值（亿），默认 200.0")
    parser.add_argument("--min-volume-ratio", type=float, default=2.0, help="最小量比，默认 2.0")
    parser.add_argument("--max-volume-ratio", type=float, default=4.0, help="最大量比，默认 4.0")
    parser.add_argument("--min-turnover", type=float, default=5.0, help="最小换手率（%%），默认 5.0")
    parser.add_argument("--max-turnover", type=float, default=10.0, help="最大换手率（%%），默认 10.0")
    parser.add_argument("--no-sector-filter", action="store_true", help="禁用板块资金净流入筛选")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("股票筛选条件：")
    logger.info(f"  涨幅: {args.min_change}% - {args.max_change}%")
    logger.info(f"  市值: {args.min_cap} - {args.max_cap} 亿")
    logger.info(f"  量比: {args.min_volume_ratio} - {args.max_volume_ratio}")
    logger.info(f"  换手率: {args.min_turnover}% - {args.max_turnover}%")
    logger.info(f"  板块资金净流入: {not args.no_sector_filter}")
    logger.info("=" * 60)

    result = filter_stocks(
        min_change_pct=args.min_change,
        max_change_pct=args.max_change,
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
        min_volume_ratio=args.min_volume_ratio,
        max_volume_ratio=args.max_volume_ratio,
        min_turnover=args.min_turnover,
        max_turnover=args.max_turnover,
        sector_net_inflow_only=not args.no_sector_filter,
    )

    print_results(result)
    return 0 if result is not None and not result.empty else 1


if __name__ == "__main__":
    sys.exit(main())
