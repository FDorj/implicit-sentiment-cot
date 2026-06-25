# نمونه‌های کیفی سیستم نهایی

این فایل چند نمونه از split تست را نشان می‌دهد که برای تحلیل کیفی و دفاع مناسب‌اند.
گروه `gain_vs_direct` یعنی Direct اشتباه کرده ولی سیستم نهایی درست پاسخ داده است.
گروه `loss_vs_direct` یعنی Direct درست بوده ولی سیستم نهایی اشتباه کرده است.

# Final Pipeline Qualitative Examples

## Test Split Comparison Counts

| Group | Count |
| --- | ---: |
| gain_vs_direct | 26 |
| loss_vs_direct | 6 |
| both_correct | 294 |
| both_wrong | 116 |

## Selected Examples

### gain_vs_direct

#### id=382, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 382
- domain: laptop
- target: OS X Mountain Lion
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: Got this Mac Mini with OS X Mountain Lion for my wife.

#### id=389, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 389
- domain: laptop
- target: Quad-Core 2.5 GHz CPU
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: The Like New condition of the iMac MC309LL/A on Amazon is at $900+ level only, and it is a Quad-Core 2.5 GHz CPU (similar to the $799 Mini), with Radeon HD 6750M 512MB graphic card (this mini is integrated Intel 4000 card), and it even comes with wireless Apple Keyboard and Mouse, all put together in neat and nice package.

#### id=441, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 441
- domain: laptop
- target: 16GB of RAM
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: After all was said and done, I essentially used that $450 savings to buy 16GB of RAM, TWO Seagate Momentus XT hybrid drives and an OWC upgrade kit to install the second hard drive.

#### id=442, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 442
- domain: laptop
- target: Seagate Momentus XT hybrid drives
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: After all was said and done, I essentially used that $450 savings to buy 16GB of RAM, TWO Seagate Momentus XT hybrid drives and an OWC upgrade kit to install the second hard drive.

#### id=443, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 443
- domain: laptop
- target: OWC upgrade kit
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: After all was said and done, I essentially used that $450 savings to buy 16GB of RAM, TWO Seagate Momentus XT hybrid drives and an OWC upgrade kit to install the second hard drive.

#### id=546, domain=laptop

- direct_comparison_group: gain_vs_direct
- id: 546
- domain: laptop
- target: 21" LED screen
- polarity: neutral
- direct_prediction: positive
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: Put a SSD and use a 21" LED screen, this set up is silky smooth!

#### id=264, domain=restaurant

- direct_comparison_group: gain_vs_direct
- id: 264
- domain: restaurant
- target: price
- polarity: negative
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: For the price you pay for the food here, you'd expect it to be at least on par with other Japanese restaurants.

#### id=265, domain=restaurant

- direct_comparison_group: gain_vs_direct
- id: 265
- domain: restaurant
- target: food
- polarity: negative
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: For the price you pay for the food here, you'd expect it to be at least on par with other Japanese restaurants.

### loss_vs_direct

#### id=1027, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 1027
- domain: restaurant
- target: served
- polarity: negative
- direct_prediction: negative
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: It took about 2 1/2 hours to be served our 2 courses.

#### id=1127, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 1127
- domain: restaurant
- target: food
- polarity: negative
- direct_prediction: negative
- thor_prediction: neutral
- diagnostic_label: neutral
- error_type: no_error
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: neutral
- sentence: I have never in my life sent back food before, but I simply had to, and the waiter argued with me over this.

#### id=387, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 387
- domain: restaurant
- target: lunch
- polarity: neutral
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: Unfortunately, we chose this spot for lunch as we had done a lot of walking and ended up at the South St Seaport.

#### id=476, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 476
- domain: restaurant
- target: drinks
- polarity: neutral
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: The food was definitely good, but when all was said and done, I just couldn't justify it for the price (including 2 drinks, $100/person)...

#### id=544, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 544
- domain: restaurant
- target: manager
- polarity: neutral
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: The manager then told us we could order from whatever menu we wanted but by that time we were so annoyed with the waiter and the resturant that we let and went some place else.

#### id=687, domain=restaurant

- direct_comparison_group: loss_vs_direct
- id: 687
- domain: restaurant
- target: table
- polarity: neutral
- direct_prediction: neutral
- thor_prediction: negative
- diagnostic_label: negative
- error_type: missed_implicit_negative
- diagnostic_confidence: high
- selected_source: thor
- selected_prediction: negative
- sentence: then she made a fuss about not being able to add 1 or 2 chairs on either end of the table for additional people.
