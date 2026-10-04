// The user must define CLA_C in the project linker settings if using the
// CLA C compiler
// Project Properties -> C2000 Linker -> Advanced Options -> Command File
// Preprocessing -> --define
#ifdef CLA_C
// Define a size for the CLA scratchpad area that will be used
// by the CLA compiler for local symbols and temps
// Also force references to the special symbols that mark the
// scratchpad are.
CLA_SCRATCHPAD_SIZE = 0x100;
--undef_sym=__cla_scratchpad_end
--undef_sym=__cla_scratchpad_start
#endif //CLA_C

MEMORY
{
PAGE 0 :
   /* BEGIN is used for the "boot to SARAM" bootloader mode   */

   BEGIN           	: origin = 0x080000,   length = 0x000002
   RAMM0           	: origin = 0x000123,   length = 0x0002DD
   RAMD0           	: origin = 0x00B000,   length = 0x000800
   //RAMLS0          	: origin = 0x008000,   length = 0x000800
   //RAMLS1          	: origin = 0x008800,   length = 0x000800
   //RAMLS_0_1_2      : origin = 0x008000, length = 0x001800
   //RAMLS_0_1_2_3    : origin = 0x008000, length = 0x002000
   RAMLS_0_1_2_3_4  : origin = 0x008000, length = 0x002800
   /* RAMLS4      	   : origin = 0x00A000, length = 0x000800 */
   /* RAMLS5           : origin = 0x00A800, length = 0x000800 */
   //RAMLS4_5         : origin = 0x00A000,   length = 0x001000
	RAMLS5_PROG      : origin = 0x00A800, length = 0x000800
   //RAMGS14          : origin = 0x01A000,   length = 0x001000     /* Only Available on F28379D, F28377D, F28375D devices. Remove line on other devices. */
   //RAMGS15          : origin = 0x01B000, length = 0x000FF8     /* Only Available on F28379D, F28377D, F28375D devices. Remove line on other devices. */

//   RAMGS15_RSVD     : origin = 0x01BFF8, length = 0x000008    /* Reserve and do not use for code as per the errata advisory "Memory: Prefetching Beyond Valid Memory" */

   RESET           	: origin = 0x3FFFC0,   length = 0x000002

   /* Flash sectors */
   FLASHA           : origin = 0x080002,   length = 0x001FFE	/* on-chip Flash */
   FLASHB           : origin = 0x082000,   length = 0x004000	/* on-chip Flash */
   //FLASHC           : origin = 0x084000,   length = 0x002000	/* on-chip Flash */
   FLASHD           : origin = 0x086000,   length = 0x002000	/* on-chip Flash */
   // GUNCELLEME: FLASHE/F/G/H fiziksel olarak bitisik (0x088000-0x0A8000)
   // ve hicbiri kullanilmiyordu -- TEK bir 0x20000 (128KB) bolge olarak
   // birlestirildi. Sebep: uc golden_adc dizisinin her biri 0x8400 (33792)
   // byte -- tek bir 0x8000 (32768) byte'lik bankaya SIGMIYOR (tam 1KB
   // eksik kaliyordu). Birlestirilmis bolge, hicbirini boyutsal olarak
   // asmadan hepsini rahatlikla barindiriyor.
   FLASH_EFGH       : origin = 0x088000,   length = 0x020000	/* on-chip Flash (E+F+G+H birlesik) */
   FLASHI           : origin = 0x0A8000,   length = 0x008000	/* on-chip Flash */
   FLASHJ           : origin = 0x0B0000,   length = 0x008000	/* on-chip Flash */
   FLASHK           : origin = 0x0B8000,   length = 0x002000	/* on-chip Flash */
   FLASHL           : origin = 0x0BA000,   length = 0x002000	/* on-chip Flash */
   FLASHM           : origin = 0x0BC000,   length = 0x002000	/* on-chip Flash */
   FLASHN           : origin = 0x0BE000, length = 0x001FF0	/* on-chip Flash */

//   FLASHN_RSVD     : origin = 0x0BFFF0, length = 0x000010    /* Reserve and do not use for code as per the errata advisory "Memory: Prefetching Beyond Valid Memory" */


PAGE 1 :

   BOOT_RSVD       : origin = 0x000002, length = 0x000121     /* Part of M0, BOOT rom will use this for stack */
   RAMM1           : origin = 0x000400, length = 0x0003F8     /* on-chip RAM block M1 */
//   RAMM1_RSVD      : origin = 0x0007F8, length = 0x000008     /* Reserve and do not use for code as per the errata advisory "Memory: Prefetching Beyond Valid Memory" */

   //RAMLS2      		: origin = 0x009000,   length = 0x000800
   //RAMLS3      		: origin = 0x009800,   length = 0x000800

   //RAMGS0           : origin = 0x00C000,   length = 0x001000
   //RAMGS1           : origin = 0x00D000,   length = 0x001000
   RAMGS_INPUT_OP  : origin = 0x00C000, length = 0x002000
   //RAMGS2           : origin = 0x00E000,   length = 0x001000
   //RAMGS3           : origin = 0x00F000,   length = 0x001000
   //RAMGS4           : origin = 0x010000,   length = 0x001000
   RAMGS_234   : origin = 0x00E000,   length = 0x003000
   //RAMGS5           : origin = 0x011000,   length = 0x001000
   //RAMGS6           : origin = 0x012000,   length = 0x001000
   //RAMGS7           : origin = 0x013000,   length = 0x001000
   //RAMGS8           : origin = 0x014000,   length = 0x001000
   //RAMGS9           : origin = 0x015000,   length = 0x001000
   //RAMGS10          : origin = 0x016000,   length = 0x001000
   //RAMGS_7_8_9_10     : origin = 0x013000,   length = 0x004000

//   RAMGS11          : origin = 0x017000, length = 0x000FF8   /* Uncomment for F28374D, F28376D devices */

//   RAMGS11_RSVD     : origin = 0x017FF8, length = 0x000008    /* Reserve and do not use for code as per the errata advisory "Memory: Prefetching Beyond Valid Memory" */

   //RAMGS11          : origin = 0x017000, length = 0x001000     /* Only Available on F28379D, F28377D, F28375D devices. Remove line on other devices. */
   //RAMGS12          : origin = 0x018000, length = 0x001000     /* Only Available on F28379D, F28377D, F28375D devices. Remove line on other devices. */
   //RAMGS_EBSS_AREA  : origin = 0x011000, length = 0x008000
   //RAMGS13          : origin = 0x019000,   length = 0x001000     /* Only Available on F28379D, F28377D, F28375D devices. Remove line on other devices. */
	RAMGS_TOTAL_CNN  : origin = 0x011000, length = 0x00B000

   EMIF1_CS0n       : origin = 0x80000000, length = 0x10000000
   EMIF1_CS2n       : origin = 0x00100000, length = 0x00200000
   EMIF1_CS3n       : origin = 0x00300000, length = 0x00080000
   EMIF1_CS4n       : origin = 0x00380000, length = 0x00060000
   EMIF2_CS0n       : origin = 0x90000000, length = 0x10000000
   EMIF2_CS2n       : origin = 0x00002000, length = 0x00001000

   CLA1_MSGRAMLOW   : origin = 0x001480,   length = 0x000080
   CLA1_MSGRAMHIGH  : origin = 0x001500,   length = 0x000080
}


SECTIONS
{
   /* Allocate program areas: */
   .cinit           : > FLASHB      PAGE = 0, ALIGN(8)
   .text            : > FLASHB|FLASHD      PAGE = 0, ALIGN(8)
   codestart        : > BEGIN       PAGE = 0, ALIGN(8)
   .stack           : > RAMM1       PAGE = 1
   .switch          : > FLASHB|FLASHD|FLASH_EFGH      PAGE = 0, ALIGN(8)

   /* Allocate uninitalized data sections: */

#if defined(__TI_EABI__)
   .init_array         : > FLASHB,       PAGE = 0, ALIGN(8)
   .bss                : > RAMGS_TOTAL_CNN,       PAGE = 1
   .bss:output         : > RAMGS_TOTAL_CNN,       PAGE = 1
   .data               : > RAMGS_TOTAL_CNN,       PAGE = 1
   .sysmem             : > RAMGS_TOTAL_CNN,       PAGE = 1
   .const              : > FLASHB,       PAGE = 0, ALIGN(8)
#else
   .pinit              : > FLASHB,       PAGE = 0, ALIGN(8)
   .ebss               : > RAMGS_TOTAL_CNN, /*RAMLS3*/		 PAGE = 1
   .esysmem            : > RAMGS_TOTAL_CNN,       PAGE = 1
   // GUNCELLEME: 3 sinifin golden_adc'i (her biri 16896 uint16 = 33792 byte,
   // toplam ~101KB) eskiden FLASHB|FLASHD|FLASHE'ye sigmiyordu (ayrica tek
   // bir dizi tek bir bankaya da sigmiyordu, bkz. FLASH_EFGH notu yukarida).
   // Simdi birlesik FLASH_EFGH (128KB) hedef listesine eklendi.
   .econst             : > FLASHB|FLASHD|FLASH_EFGH   	 PAGE = 0, ALIGN(8)
#endif

   .reset           : > RESET,     PAGE = 0, TYPE = DSECT /* not used, */

   Filter_RegsFile  : > RAMGS_234,	   PAGE = 1

   .em2_cs0         : > EMIF2_CS0n, PAGE = 1
   .em2_cs2         : > EMIF2_CS2n, PAGE = 1
   ramgs0           : > RAMGS_234,     PAGE = 1

    /* CLA specific sections */
   #if defined(__TI_EABI__)
   		Cla1Prog    : LOAD = FLASHD,
                      RUN = RAMLS5_PROG,
                      LOAD_START(Cla1funcsLoadStart),
                      LOAD_END(Cla1funcsLoadEnd),
                      RUN_START(Cla1funcsRunStart),
                      LOAD_SIZE(Cla1funcsLoadSize),
                      PAGE = 0, ALIGN(8)
   #else
      	Cla1Prog    : LOAD = FLASHD,
                      RUN = RAMLS5_PROG,
                      LOAD_START(_Cla1funcsLoadStart),
                      LOAD_END(_Cla1funcsLoadEnd),
                      RUN_START(_Cla1funcsRunStart),
                      LOAD_SIZE(_Cla1funcsLoadSize),
                      PAGE = 0, ALIGN(8)
   #endif

   CLADataLS0		: > RAMLS_0_1_2_3_4, PAGE=0

   /* DUZELTME (soguk acilis): CLADataLS1, CLA'nin okudugu const ilk degerli
      agirliklari tutuyor (weight[], conv1/conv2 agirlik+bias, fc_bias). Eskiden
      sadece RUN adresi (RAM) vardi -- icerigi yalnizca debugger program
      yuklerken RAM'e yaziyordu, gercek guc verme aninda RAM bos/belirsizdi.
      Kartta olculdu: RAM silinip 0x80000'den baslatilinca 111 konfigurasyonu
      tum logitleri 0 uretti (12/36 dogru), agirliklari flash'ta olan 000
      etkilenmedi. Cla1Prog ile ayni desen: flash'a yukle, acilista
      CLA_configClaMemory() RAM'e kopyalasin. (CLADataLS0 const olmayan
      tampondur; ilk degerleri .cinit ile zaten tasiniyor.) */
   #if defined(__TI_EABI__)
   		CLADataLS1  : LOAD = FLASHB,
                      RUN = RAMLS_0_1_2_3_4,
                      LOAD_START(Cla1DataLoadStart),
                      RUN_START(Cla1DataRunStart),
                      LOAD_SIZE(Cla1DataLoadSize),
                      PAGE = 0, ALIGN(8)
   #else
   		CLADataLS1  : LOAD = FLASHB,
                      RUN = RAMLS_0_1_2_3_4,
                      LOAD_START(_Cla1DataLoadStart),
                      RUN_START(_Cla1DataRunStart),
                      LOAD_SIZE(_Cla1DataLoadSize),
                      PAGE = 0, ALIGN(8)
   #endif

   Cla1ToCpuMsgRAM  : > CLA1_MSGRAMLOW,   PAGE = 1
   CpuToCla1MsgRAM  : > CLA1_MSGRAMHIGH,  PAGE = 1

#ifdef __TI_COMPILER_VERSION__
   #if __TI_COMPILER_VERSION__ >= 15009000
        #if defined(__TI_EABI__)
		    .TI.ramfunc : {} LOAD = FLASHD,
							 RUN = RAMD0,
                	         LOAD_START(RamfuncsLoadStart),
                 	         LOAD_SIZE(RamfuncsLoadSize),
                 	         LOAD_END(RamfuncsLoadEnd),
                  	         RUN_START(RamfuncsRunStart),
                  	         RUN_SIZE(RamfuncsRunSize),
                   	         RUN_END(RamfuncsRunEnd),
							 PAGE = 0, ALIGN(8)
		#else
			.TI.ramfunc : {} LOAD = FLASHD,
							 RUN = RAMD0,
                	         LOAD_START(_RamfuncsLoadStart),
                 	         LOAD_SIZE(_RamfuncsLoadSize),
                 	         LOAD_END(_RamfuncsLoadEnd),
                  	         RUN_START(_RamfuncsRunStart),
                  	         RUN_SIZE(_RamfuncsRunSize),
                   	         RUN_END(_RamfuncsRunEnd),
							 PAGE = 0, ALIGN(8)
		#endif
   #else
   ramfuncs         : LOAD = FLASHD,
                      RUN = RAMD0,
                      LOAD_START(_RamfuncsLoadStart),
                      LOAD_SIZE(_RamfuncsLoadSize),
                      LOAD_END(_RamfuncsLoadEnd),
                      RUN_START(_RamfuncsRunStart),
                      RUN_SIZE(_RamfuncsRunSize),
                      RUN_END(_RamfuncsRunEnd),
                      PAGE = 0, ALIGN(8)
   #endif
#endif

   /* The following section definition are for SDFM examples */
   //Filter1_RegsFile : > RAMGS2,	PAGE = 1, fill=0x1111
   //Filter2_RegsFile : > RAMGS2,	PAGE = 1, fill=0x2222
   //Filter3_RegsFile : > RAMGS3,	PAGE = 1, fill=0x3333
   //Filter4_RegsFile : > RAMGS4,	PAGE = 1, fill=0x4444

#ifdef CLA_C
   /* CLA C compiler sections */
   //
   // Must be allocated to memory the CLA has write access to
   //
   CLAscratch       :
                     { *.obj(CLAscratch)
                     . += CLA_SCRATCHPAD_SIZE;
                     *.obj(CLAscratch_end) } >  RAMLS_0_1_2_3_4,  PAGE = 0

   .scratchpad      : > RAMLS_0_1_2_3_4,       PAGE = 0
   .bss_cla		    : > RAMLS_0_1_2_3_4,       PAGE = 0
   #if defined(__TI_EABI__)
   .const_cla      : LOAD = FLASHB,
                      RUN = RAMLS_0_1_2_3_4,
                      RUN_START(Cla1ConstRunStart),
                      LOAD_START(Cla1ConstLoadStart),
                      LOAD_SIZE(Cla1ConstLoadSize),
                      PAGE = 0,
                      ALIGN(4)
  #else
   .const_cla      : LOAD = FLASHB,
                      RUN = RAMLS_0_1_2_3_4,
                      RUN_START(_Cla1ConstRunStart),
                      LOAD_START(_Cla1ConstLoadStart),
                      LOAD_SIZE(_Cla1ConstLoadSize),
                      PAGE = 0,
                      ALIGN(4)
  #endif
#endif //CLA_C
}

/*
//===========================================================================
// End of file.
//===========================================================================
*/
