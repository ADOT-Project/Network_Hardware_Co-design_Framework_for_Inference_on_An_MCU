#ifndef _CLA_SQRT_SHARED_H_
#define _CLA_SQRT_SHARED_H_
//#############################################################################
//
// FILE:   cla_sqrt_shared.h
//
// TITLE:  Square Root Test
//
//#############################################################################
//
// 
// $Copyright:
// Copyright (C) 2013-2024 Texas Instruments Incorporated - http://www.ti.com/
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions 
// are met:
// 
//   Redistributions of source code must retain the above copyright 
//   notice, this list of conditions and the following disclaimer.
// 
//   Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the 
//   documentation and/or other materials provided with the   
//   distribution.
// 
//   Neither the name of Texas Instruments Incorporated nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// $
//#############################################################################

//
// Included Files
//
#include "F2837xD_Cla_defines.h"
#include <stdint.h>
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Globals
//

//
//Task 1 (C) Variables
//
extern uint16_t cla_count_param; //input
extern uint16_t cla_offset_param;
// GUNCELLEME: fResult[10] (MNIST'in 10 rakami) -> fResult[NUM_CLASSES] (=3).
extern float fResult[NUM_CLASSES];  //Estimated result
//extern float input_vector[DENSE_LAYER_INPUT];
extern int16_t input_vector[DENSE_LAYER_INPUT];
//extern float weight [DENSE_LAYER_INPUT];
extern const int16_t weight [DENSE_LAYER_WEIGHTS];
// EKLENDI: dense/FC katmaninin bias'i -- Cla1Task1 (dense_layer.cla) eskiden
// bias hic eklemiyordu (sum = W*X, +bias yok). fc_bias degerleri (weights.h)
// sinif ayrimini gozle gorulur sekilde etkileyecek buyuklukte oldugu icin
// (bkz. model_weights_int16.h: fc_bias=[-32767,27600,16493]) eklendi.
extern const int16_t fc_bias [NUM_CLASSES];
extern float weight_1 [200];
extern float weight_2 [200];

extern int16_t input_image[];
extern const int16_t conv1_weights[];
extern const int16_t conv1_bias[];
extern const int16_t conv2_weights[];
extern const int16_t conv2_bias[];
extern int16_t workspace[];


//
//Task 2 (C) Variables
//

//
//Task 3 (C) Variables
//

//
//Task 4 (C) Variables
//

//
//Task 5 (C) Variables
//

//
//Task 6 (C) Variables
//

//
//Task 7 (C) Variables
//

//
//Task 8 (C) Variables
//

//
//Common (C) Variables
//

//
// Function Prototypes
//
// The following are symbols defined in the CLA assembly code
// Including them in the shared header file makes them
// .global and the main CPU can make use of them.
//
__interrupt void Cla1Task1();
// GUNCELLEME: Cla1Task2..8 bildirimleri kaldirildi -- tanimlari
// dense_layer.cla'dan silindi (kullanilmiyorlardi, bkz. oradaki not).
//
// GUNCELLEME (asama bazli bolusum): karma CPU/CLA dagiliminda S2 ve S3 kendi
// gorevlerinde calisiyor (Task2=conv2+pool2, Task3=dense; bkz.
// cla_pipeline_config.h). Prototipler yalnizca gorev GERCEKTEN derlendiginde
// bildiriliyor -- bildirilmis ama tanimlanmamis bir gorevin adresi yanlislikla
// MVECT'e yazilirsa link hatasi yerine derleme hatasi alinsin diye.
// Bu blok olmadan main.c'deki Cla1Regs.MVECT2/3 atamasi TI derleyicisinde
// "identifier Cla1Task2 is undefined" (error #20) ile duser.
#include "cla_pipeline_config.h"
#if !CLA_SINGLE_TASK && CLA_STAGE_CONV2_POOL2
__interrupt void Cla1Task2();
#endif
#if !CLA_SINGLE_TASK && CLA_STAGE_DENSE
__interrupt void Cla1Task3();
#endif

#ifdef __cplusplus
}
#endif // extern "C"

#endif //end of _CLA_SQRT_SHARED_H_ definition

//
// End of file
//
