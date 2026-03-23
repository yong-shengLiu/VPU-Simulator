import re
import sys

# CSR 位址對應名稱
CSR_NAMES = {
    0x801: "CSR_VPU_MEM_BASE_A",
    0x802: "CSR_VPU_MEM_BASE_B",
    0x803: "CSR_VPU_MEM_BASE_C",
    0x804: "CSR_VPU_MEM_BASE_D",
    0x805: "CSR_VPU_MEM_STRIDE",
    0x806: "CSR_VPU_MEM_ACCESS_CFG",
    0x807: "CSR_VPU_REG_BASE_CFG",
    0x808: "CSR_VPU_STRIDE_CFG",
    0x809: "CSR_VPU_TILE_CFG",
    0x80A: "CSR_VPU_MACRO_TRIGGER"
}

def parse_log(filepath, is_rtl=False):
    """
    解析 Log 檔，回傳 List。
    內部結構: [{"name": "L0_PROJ_Q_m0_n0", "csrs": {0x801: val, ...}}, ...]
    """
    macros = []
    current_macro_csrs = {}
    current_macro_name = "UNKNOWN_MACRO"

    if is_rtl:
        # 匹配 C Code (RTL) 格式
        pattern = re.compile(r"csrw\s+.*\((0x[0-9a-fA-F]+)\)\s+<-\s+(0x[0-9a-fA-F]+)")
    else:
        # 匹配 Python (Golden) 格式
        pattern = re.compile(r"csrw\s+(0x[0-9a-fA-F]+),\s+(0x[0-9a-fA-F]+)")
        # ★ 新增：匹配 Python Log 中的 Macro 名稱
        name_pattern = re.compile(r"# --- Dispatching Macro:\s+([^\s]+)")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 如果是 Golden Log，抓取名稱
                if not is_rtl:
                    name_match = name_pattern.search(line)
                    if name_match:
                        current_macro_name = name_match.group(1)

                # 抓取 csrw 寫入動作
                match = pattern.search(line)
                if match:
                    addr = int(match.group(1), 16)
                    val = int(match.group(2), 16)

                    current_macro_csrs[addr] = val

                    # 0x80A 是 Trigger，代表一個 Macro 的結束，打包存入
                    if addr == 0x80A:
                        macros.append({
                            "name": current_macro_name,
                            "csrs": current_macro_csrs
                        })
                        current_macro_csrs = {}
                        current_macro_name = "UNKNOWN_MACRO"
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {filepath}")
        sys.exit(1)

    return macros

def compare_macros(golden_file, rtl_file):
    print("="*60)
    print("🔍 啟動 Macro CSR 比對程式")
    print("="*60)

    golden_macros = parse_log(golden_file, is_rtl=False)
    rtl_macros = parse_log(rtl_file, is_rtl=True)

    print(f"📄 讀取 Golden 巨集數量: {len(golden_macros)}")
    print(f"📄 讀取 RTL 巨集數量   : {len(rtl_macros)}")
    
    if len(golden_macros) != len(rtl_macros):
        print(f"⚠️ [警告] 巨集總數不一致！將比對前 {min(len(golden_macros), len(rtl_macros))} 個巨集...\n")

    min_len = min(len(golden_macros), len(rtl_macros))
    total_count    = 0
    mismatch_count = 0

    for i in range(min_len):
        gm_data = golden_macros[i]
        rm_data = rtl_macros[i]
        
        gm_csrs = gm_data["csrs"]
        rm_csrs = rm_data["csrs"]
        macro_name = gm_data["name"] # ★ 從 Golden 取出名稱

        for addr in range(0x801, 0x80B): # 0x801 ~ 0x80A
            g_val = gm_csrs.get(addr, None)
            r_val = rm_csrs.get(addr, None)
            total_count +=1

            if g_val != r_val:
                if mismatch_count == 0:
                    print("❌ 發現不一致的 CSR 設定：\n")
                
                csr_name = CSR_NAMES.get(addr, f"UNKNOWN_CSR")
                # ★ 顯示格式更新：加上 macro_name
                print(f"  ➤ Macro #{i} [{macro_name}] | {csr_name} (0x{addr:03X}):")
                print(f"     - Golden : {f'0x{g_val:016X}' if g_val is not None else 'Missing'}")
                print(f"     - RTL    : {f'0x{r_val:016X}' if r_val is not None else 'Missing'}")
                print("-" * 60)
                mismatch_count += 1

    print("="*60)
    if mismatch_count == 0 and len(golden_macros) == len(rtl_macros):
        print("✅ 完美對齊！Golden 與 RTL 的所有 Macro 參數完全一致！")
    elif mismatch_count == 0:
        print("⚠️ 參數皆一致，但是巨集「總數量」不同，請檢查迴圈或網路層數！")
    else:
        print(f"❌ 總共發現 {mismatch_count}/{total_count} 處不一致，請參考上方 Log 進行修正。")

if __name__ == "__main__":
    golden_log = "BERT_Base_csr_trace.txt"
    rtl_log = "check_fuck.log"
    compare_macros(golden_log, rtl_log)