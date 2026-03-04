// =========================================================
// ADHD VPU Firmware Dispatcher (TEST_SUBOP)
// =========================================================
#include <stdint.h>

static inline void dispatch_test_subop_macros() {

    // --- Dispatching Macro: Attention ---
    __asm__ volatile("csrw 0x801, %0" :: "r"(0x00000000E0000000ULL));
    __asm__ volatile("csrw 0x802, %0" :: "r"(0x00000000E0008000ULL));
    __asm__ volatile("csrw 0x803, %0" :: "r"(0x00000000E0018000ULL));
    __asm__ volatile("csrw 0x804, %0" :: "r"(0x00000000E0010000ULL));
    __asm__ volatile("csrw 0x805, %0" :: "r"(0x0300030003000300ULL));
    __asm__ volatile("csrw 0x806, %0" :: "r"(0x0000008210420841ULL));
    __asm__ volatile("csrw 0x807, %0" :: "r"(0x00000000F8A34040ULL));
    __asm__ volatile("csrw 0x808, %0" :: "r"(0x0000000010111042ULL));
    __asm__ volatile("csrw 0x809, %0" :: "r"(0x0000002000200020ULL));
    __asm__ volatile("csrw 0x80A, %0" :: "r"(0x0000000000000003ULL));
