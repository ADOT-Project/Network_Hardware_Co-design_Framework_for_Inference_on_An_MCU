/* PC stub: just enough of the device header for dense_layer.cla to compile
 * with gcc. Not used on the target. */
#ifndef PC_STUB_F2837XD_DEVICE_H
#define PC_STUB_F2837XD_DEVICE_H
struct pc_stub_gpio_bits { unsigned GPIO67; };
struct pc_stub_gpio_reg  { struct pc_stub_gpio_bits bit; };
struct pc_stub_gpio_data { struct pc_stub_gpio_reg GPCCLEAR, GPCSET; };
extern struct pc_stub_gpio_data GpioDataRegs;
#endif
