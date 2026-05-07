"use strict";
cc._RF.push(module, '895e44MoOdMzJ2X1sOV6aHH', 'global - 002');
// 3.16小游戏/command_TypeScript/level6/global - 002.ts

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
var global = /** @class */ (function (_super) {
    __extends(global, _super);
    function global() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    global_1 = global;
    global.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    global.prototype.onEventStart = function () {
        global_1.attack = true;
    };
    global.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + global_1.minion1_hp + " " + global_1.minion3_hp + " " + global_1.minion3_hp);
    };
    var global_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    global.attack = false;
    global.minion_attack = false;
    global.main_hp = 200;
    global.minion1_hp = 400;
    global.minion2_hp = 200;
    global.minion3_hp = 200;
    __decorate([
        property(cc.Label)
    ], global.prototype, "label", void 0);
    __decorate([
        property
    ], global.prototype, "text", void 0);
    global = global_1 = __decorate([
        ccclass
    ], global);
    return global;
}(cc.Component));
exports.default = global;

cc._RF.pop();