"use strict";
cc._RF.push(module, '91522+UAMROjo6f/8WxasmZ', 'main');
// 3.16小游戏/command_TypeScript/level2/main.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_1 = /** @class */ (function (_super) {
    __extends(main_1, _super);
    function main_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1_1 = main_1;
    main_1.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main_1.prototype.onEventStart = function () {
        main_1_1.attack = true;
    };
    main_1.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + main_1_1.minion1_hp + " " + main_1_1.minion3_hp + " " + main_1_1.minion3_hp);
    };
    var main_1_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main_1.attack = false;
    main_1.minion_attack = false;
    main_1.main_hp = 163;
    main_1.minion1_hp = 163;
    main_1.minion2_hp = 163;
    main_1.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main_1.prototype, "label", void 0);
    __decorate([
        property
    ], main_1.prototype, "text", void 0);
    main_1 = main_1_1 = __decorate([
        ccclass
    ], main_1);
    return main_1;
}(cc.Component));
exports.default = main_1;

cc._RF.pop();