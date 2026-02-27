module cim_execution_unit #(
    parameter PHYS_M = 16,
    parameter PHYS_N = 16,
    parameter PHYS_K = 16  // CIM 實體陣列一次能處理的 K 深度
)(
    input  logic clk,
    input  logic rst_n,

    // -----------------------------------------
    // 1. Issue Interface (從解耦 Queue 接收 MicroOp)
    // -----------------------------------------
    input  logic        issue_valid,
    output logic        issue_ready,
    input  logic [4:0]  src_a_reg,   // 例如 v0
    input  logic [4:0]  src_b_reg,   // 例如 v4
    input  logic [7:0]  k_tile,      // Python 裡的 csr.K_tile (例如 32)

    // -----------------------------------------
    // 2. VRF Read Interface (讀取 A 矩陣 / B矩陣)
    // -----------------------------------------
    output logic        vrf_read_req,
    output logic [9:0]  vrf_addr_a,  
    input  logic [127:0] vrf_rdata_a, // 16 bytes (1x16 INT8 Vector)

    // -----------------------------------------
    // 3. L0 Buffer Interface (累加與儲存)
    // -----------------------------------------
    output logic        l0_write_req,
    output logic [3:0]  l0_row_addr,  // 指向 0~15 列
    output logic [511:0] l0_wdata     // 16 個 FP32 (16 * 32bit = 512bit)
);

    // 狀態機定義
    typedef enum logic [1:0] {
        ST_IDLE    = 2'b00,
        ST_LOAD_W  = 2'b01, // 背景載入 Weight 到 CIM SRAM
        ST_COMPUTE = 2'b10  // 核心計算 (產生 M_tile * K_chunks 個 Cycle)
    } state_t;

    state_t state, next_state;

    // 內部計數器
    logic [3:0] m_cnt;       // 追蹤 0~15 (M 維度)
    logic [3:0] k_chunk_cnt; // 追蹤 K 維度切了幾塊 (例如 K=32，就是 2 塊)
    
    logic [3:0] num_k_chunks;
    assign num_k_chunks = k_tile / PHYS_K; // e.g., 32 / 16 = 2

    // ====================================================================
    // 🌟 FSM 狀態轉移邏輯
    // ====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            m_cnt       <= '0;
            k_chunk_cnt <= '0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (issue_valid) begin
                        state <= ST_COMPUTE; // 假設 Weight 已經透過 Shadow Buffer 載好
                        m_cnt <= '0;
                        k_chunk_cnt <= '0;
                    end
                end

                ST_COMPUTE: begin
                    // 【核心邏輯】：每個 Cycle 走一步 M
                    if (m_cnt == PHYS_M - 1) begin
                        m_cnt <= '0;
                        // M 走完了，看看 K 有沒有下一塊
                        if (k_chunk_cnt == num_k_chunks - 1) begin
                            state <= ST_IDLE; // 全部算完，回到 IDLE，釋放 Scoreboard！
                        end else begin
                            k_chunk_cnt <= k_chunk_cnt + 1;
                        end
                    end else begin
                        m_cnt <= m_cnt + 1;
                    end
                end
            endcase
        end
    end

    // ====================================================================
    // 🌟 Datapath: 每一個 Cycle 都在幹嘛？
    // ====================================================================
    
    // 1. 產生 VRF 讀取位址 (每個 Cycle 讀取 A 矩陣的下一列)
    assign vrf_read_req = (state == ST_COMPUTE);
    assign vrf_addr_a   = {src_a_reg, k_chunk_cnt, m_cnt[1:0]}; // 示意定址邏輯
    
    // 2. 模擬真實的 SRAM-CIM 巨集 (Combinational 抽象)
    logic [511:0] cim_mac_result;
    
    /* 這裡實體化真正的 SRAM-CIM IP。
       它吃 VRF 讀出來的 1x16 Vector (A)，
       跟自己 SRAM 肚子裡的 16x16 權重 (B) 做內積，
       瞬間吐出 1x16 的 FP32 Psum。
    */
    sram_cim_macro #(16, 16) u_cim_macro (
        .act_in(vrf_rdata_a),    // A 矩陣的一列 (Broadcast)
        .psum_out(cim_mac_result) // C 矩陣的一列
    );

    // 3. 寫回 L0 Psum Buffer
    assign l0_write_req = (state == ST_COMPUTE);
    assign l0_row_addr  = m_cnt;  // 寫入對應的 M 列
    
    // L0 Buffer 會做 Read-Modify-Write (累加前一次的 Psum)
    // l0_wdata = cim_mac_result + l0_rdata; (此處簡化)
    assign l0_wdata = cim_mac_result; 

    // 告訴前端我現在能不能收新指令
    assign issue_ready = (state == ST_IDLE);

endmodule