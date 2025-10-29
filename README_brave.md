# Brave 版 ESG 爬蟲使用說明

## 基本指令
```bash
python scripts/esg_crawler_brave.py --root /mnt/data/esg_pipeline   --start-year 2017 --end-year 2018   --brave-key BSAav9ydZ7i0dbLIArmfG4465srvMbB   --max-results 8 --throttle-sec 1.0
```

```bash
uv run src/esg_crawler_brave.py --start-year 2017 --end-year 2018   --brave-key BSAav9ydZ7i0dbLIArmfG4465srvMbB   --max-results 8 --throttle-sec 1.0
```
