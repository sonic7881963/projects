"use strict";
cc._RF.push(module, 'ac1f3CNYoBK/rt2tvbz7Ys1', 'global - 001');
// 3.16小游戏/command_TypeScript/level5/global - 001.ts

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
var gloabl = /** @class */ (function (_super) {
    __extends(gloabl, _super);
    function gloabl() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    gloabl_1 = gloabl;
    gloabl.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    gloabl.prototype.onEventStart = function () {
        cc.log("click");
        gloabl_1.attack = true;
    };
    var gloabl_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    gloabl.attack = false;
    gloabl.minion_attack = false;
    gloabl.main_hp = 163;
    gloabl.minion1_hp = 280;
    gloabl.minion2_hp = 163;
    gloabl.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], gloabl.prototype, "label", void 0);
    __decorate([
        property
    ], gloabl.prototype, "text", void 0);
    gloabl = gloabl_1 = __decorate([
        ccclass
    ], gloabl);
    return gloabl;
}(cc.Component));
exports.default = gloabl;

cc._RF.pop();